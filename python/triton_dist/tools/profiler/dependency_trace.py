################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Iterable

from triton_dist.models.utils import logger as mega_logger

import torch

from .viewer import parse_to_tracks, Task as ProfiledTask

from triton_dist.mega_triton_kernel.core.task_base import TaskBase, TaskDependency


@dataclass
class TraceTask:
    node_id: str
    task_type: str
    task_type_id: int
    layer_id: int
    task_id: int
    tile_id: int
    sm_id: int
    start_time_ns: int
    duration_ns: int
    finish_time_ns: int
    absolute_start_time_ns: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceDependency:
    src: str
    dst: str
    start_tile: int
    end_tile: int
    origin: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CriticalPath:
    total_duration_ns: int
    slack_ns: int
    tasks: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DependencyTraceBuilder:
    def __init__(
        self,
        profiler_buffer: torch.Tensor,
        task_types_to_str: Dict[int, str],
        wq_tensor: torch.Tensor,
        num_tasks_tensor: torch.Tensor,
        scheduled_tasks: Iterable["TaskBase"],
        base_dir: Optional[str] = None,
    ) -> None:
        self._profiler_buffer = profiler_buffer
        self._task_types_to_str = task_types_to_str
        self._wq_tensor = wq_tensor
        self._num_tasks_tensor = num_tasks_tensor
        self._scheduled_tasks = list(scheduled_tasks)
        self._task_lookup: Dict[Tuple[int, int, int, int], TaskBase] = {}
        self._task_nodes: Dict[str, TraceTask] = {}
        self._edges: List[TraceDependency] = []
        self._graph: Dict[str, List[str]] = {}
        self._reverse_graph: Dict[str, List[str]] = {}
        self._path_finish_time: Dict[str, int] = {}
        self._pred: Dict[str, Optional[str]] = {}
        self._min_start_time: int = 0
        self._logical_to_key: Dict[Tuple[int, int, int], List[Tuple[int, int, int, int]]] = {}
        self._base_dir = base_dir

    # ---------------------------- helpers ---------------------------------
    def _build_task_lookup(self) -> None:
        for task in self._scheduled_tasks:
            assert isinstance(task, TaskBase)
            key = (task.get_task_type_id(), task.layer_id, task.task_id, task.tile_id_or_start)
            if key in self._task_lookup:
                raise RuntimeError(
                    f"Duplicate task key {key} encountered when building dependency trace."
                )
            self._task_lookup[key] = task
            logical_key = (task.layer_id, task.task_id, task.tile_id_or_start)
            self._logical_to_key.setdefault(logical_key, []).append(key)

    def _decode_work_queues(self) -> Dict[int, List[Tuple[int, int, int, int]]]:
        wq_host = self._wq_tensor.cpu()
        num_tasks_host = self._num_tasks_tensor.cpu()
        max_tasks, num_sms, _ = wq_host.shape
        decoded: Dict[int, List[Tuple[int, int, int, int]]] = {}
        padding_value = (1 << 32) - 1  # uint32 representation of -1
        for sm_id in range(num_sms):
            tasks = []
            num_tasks = int(num_tasks_host[sm_id].item())
            for task_idx in range(min(num_tasks, max_tasks)):
                entry = wq_host[task_idx, sm_id]
                task_type_id = int(entry[0].item())
                if task_type_id == padding_value:
                    continue
                layer_id = int(entry[1].item())
                task_id = int(entry[2].item())
                tile_id = int(entry[3].item())
                tasks.append((task_type_id, layer_id, task_id, tile_id))
            decoded[sm_id] = tasks
        return decoded

    def _collect_profiled_events(self) -> Dict[int, List[ProfiledTask]]:
        block_tracks = parse_to_tracks(self._profiler_buffer)
        filtered: Dict[int, List[ProfiledTask]] = {}
        skip_types = {
            k for k, v in self._task_types_to_str.items()
            if v in {"scoreboard_wait_deps", "task_decoding"}
        }
        for block_idx, tasks in block_tracks.items():
            filtered_tasks = [t for t in tasks if t.task_type not in skip_types]
            filtered_tasks.sort(key=lambda t: t.start_time)
            filtered[block_idx] = filtered_tasks
        return filtered

    def _create_node_id(self, task_type_id: int, layer_id: int, task_id: int, tile_id: int) -> str:
        return f"{task_type_id}:{layer_id}:{task_id}:{tile_id}"

    def _build_nodes(self) -> None:
        decoded_wq = self._decode_work_queues()
        profiled_events = self._collect_profiled_events()
        all_start_times: List[int] = []
        for sm_id, scheduled in decoded_wq.items():
            events = profiled_events.get(sm_id, [])
            if len(events) != len(scheduled):
                mega_logger.log(
                    (
                        "Profiler events (%d) did not match scheduled tasks (%d) for SM %d. "
                        "Attempting best-effort alignment."
                    )
                    % (len(events), len(scheduled), sm_id),
                    level="warning",
                )
            event_idx = 0
            for entry in scheduled:
                key = entry
                task = self._task_lookup.get(key, None)
                if task is None:
                    mega_logger.log(
                        f"Unable to map scheduled task {key} to TaskBase instance; skipping.",
                        level="warning",
                    )
                    continue
                event = None
                while event_idx < len(events):
                    candidate = events[event_idx]
                    event_idx += 1
                    if candidate.task_type == key[0]:
                        event = candidate
                        break
                    mega_logger.log(
                        (
                            "Dropping profiler event with task_type %d while aligning SM %d "
                            "because it does not match scheduled task %s."
                        )
                        % (candidate.task_type, sm_id, key),
                        level="warning",
                    )
                if event is None:
                    mega_logger.log(
                        f"No profiler event available for scheduled task {key} on SM {sm_id}; skipping node.",
                        level="warning",
                    )
                    continue
                node_id = self._create_node_id(*key)
                start_time_ns = int(event.start_time)
                duration_ns = int(event.duration)
                all_start_times.append(start_time_ns)
                finish_time_ns = start_time_ns + duration_ns
                self._task_nodes[node_id] = TraceTask(
                    node_id=node_id,
                    task_type=self._task_types_to_str.get(key[0], f"task_{key[0]}"),
                    task_type_id=key[0],
                    layer_id=key[1],
                    task_id=key[2],
                    tile_id=key[3],
                    sm_id=sm_id,
                    start_time_ns=start_time_ns,
                    duration_ns=duration_ns,
                    finish_time_ns=finish_time_ns,
                    absolute_start_time_ns=start_time_ns,
                )
        if not all_start_times:
            raise RuntimeError(
                "Profiler trace buffer did not contain any task events after alignment; "
                "ensure profiling is enabled and the dependency trace is exported post-run."
            )
        self._min_start_time = min(all_start_times)
        for node in self._task_nodes.values():
            relative_start = node.start_time_ns - self._min_start_time
            node.start_time_ns = relative_start
            node.finish_time_ns = relative_start + node.duration_ns

    def _build_edges(self) -> None:
        adjacency: Dict[str, List[str]] = {}
        reverse_adj: Dict[str, List[str]] = {}
        edge_keys = set()
        for node_id, node in self._task_nodes.items():
            task = self._task_lookup[(node.task_type_id, node.layer_id, node.task_id, node.tile_id)]
            if not adjacency.get(node_id):
                adjacency[node_id] = []
            if node_id not in reverse_adj:
                reverse_adj[node_id] = []
            for dep in task.dependency:
                assert isinstance(dep, TaskDependency)
                producer_keys: List[Tuple[int, int, int, int]] = []
                for tile_id in range(dep.start_tiles, dep.end_tiles):
                    logical_key = (dep.layer_id, dep.task_id, tile_id)
                    producer_keys = self._logical_to_key.get(logical_key, [])
                    if producer_keys:
                        break
                if not producer_keys:
                    mega_logger.log(
                        f"Failed to find producer for dependency {dep} on task {node_id}; skipping edge.",
                        level="warning",
                    )
                    continue
                found_src = False
                for producer_key in producer_keys:
                    src_id = self._create_node_id(*producer_key)
                    if src_id not in self._task_nodes:
                        continue
                    edge_key = (src_id, node_id, dep.start_tiles, dep.end_tiles)
                    if edge_key in edge_keys:
                        found_src = True
                        break
                    edge_keys.add(edge_key)
                    origin_dict = dep.origin.to_dict() if dep.origin else None
                    if origin_dict and self._base_dir:
                        filename = origin_dict.get("filename")
                        if filename:
                            try:
                                rel = os.path.relpath(filename, start=self._base_dir)
                                if not rel.startswith(".."):
                                    origin_dict["filename"] = rel
                            except ValueError:
                                pass
                    edge = TraceDependency(
                        src=src_id,
                        dst=node_id,
                        start_tile=dep.start_tiles,
                        end_tile=dep.end_tiles,
                        origin=origin_dict,
                    )
                    self._edges.append(edge)
                    adjacency.setdefault(src_id, []).append(node_id)
                    reverse_adj.setdefault(node_id, []).append(src_id)
                    found_src = True
                    break
                if not found_src:
                    mega_logger.log(
                        f"Unable to map dependency {dep} for task {node_id} to profiled producer; skipping edge.",
                        level="warning",
                    )
        self._graph = adjacency
        self._reverse_graph = reverse_adj

    def _topological_order(self) -> List[str]:
        indegree: Dict[str, int] = {node: 0 for node in self._task_nodes}
        for src, dsts in self._graph.items():
            for dst in dsts:
                indegree[dst] = indegree.get(dst, 0) + 1
        queue = [node for node, deg in indegree.items() if deg == 0]
        topo: List[str] = []
        idx = 0
        while idx < len(queue):
            node = queue[idx]
            idx += 1
            topo.append(node)
            for nbr in self._graph.get(node, []):
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    queue.append(nbr)
        if len(topo) != len(self._task_nodes):
            raise RuntimeError("Cycle detected in dependency graph; cannot compute critical path.")
        return topo

    def _compute_critical_paths(self, max_paths: int = 5) -> List[CriticalPath]:
        topo = self._topological_order()
        finish_time: Dict[str, int] = {}
        pred: Dict[str, Optional[str]] = {node: None for node in self._task_nodes}
        for node_id in topo:
            node = self._task_nodes[node_id]
            if not self._reverse_graph.get(node_id):
                finish_time[node_id] = node.finish_time_ns
            else:
                best_finish = node.finish_time_ns
                best_pred = None
                for src in self._reverse_graph[node_id]:
                    candidate_ready = max(finish_time[src], node.start_time_ns)
                    candidate_finish = candidate_ready + node.duration_ns
                    if candidate_finish > best_finish:
                        best_finish = candidate_finish
                        best_pred = src
                finish_time[node_id] = best_finish
                pred[node_id] = best_pred
        self._path_finish_time = finish_time
        self._pred = pred
        sorted_nodes = sorted(finish_time.items(), key=lambda kv: kv[1], reverse=True)
        critical_total = sorted_nodes[0][1] if sorted_nodes else 0
        paths: List[CriticalPath] = []
        seen_paths: set = set()
        for node_id, finish in sorted_nodes[:max_paths * 2]:
            path_nodes = []
            cur = node_id
            while cur is not None:
                path_nodes.append(cur)
                cur = pred.get(cur)
            path_nodes.reverse()
            path_key = tuple(path_nodes)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            slack = max(0, critical_total - finish)
            paths.append(CriticalPath(total_duration_ns=finish, slack_ns=slack, tasks=path_nodes))
            if len(paths) >= max_paths:
                break
        return paths

    # ---------------------------- public API ---------------------------------
    def build(self, max_paths: int = 5) -> Dict[str, Any]:
        self._build_task_lookup()
        self._build_nodes()
        self._build_edges()
        critical_paths = self._compute_critical_paths(max_paths=max_paths)
        tasks_sorted = sorted(self._task_nodes.values(), key=lambda n: (n.start_time_ns, n.node_id))
        edges_sorted = sorted(
            self._edges,
            key=lambda e: (e.dst, e.src, e.start_tile, e.end_tile),
        )
        return {
            "tasks": [node.to_dict() for node in tasks_sorted],
            "dependencies": [edge.to_dict() for edge in edges_sorted],
            "critical_paths": [path.to_dict() for path in critical_paths],
            "min_start_time_ns": self._min_start_time,
        }


def export_dependency_trace(
    profiler_buffer: torch.Tensor,
    task_types_to_str: Dict[int, str],
    trace_file: str,
    wq_tensor: torch.Tensor,
    num_tasks_tensor: torch.Tensor,
    scheduled_tasks: Iterable["TaskBase"],
    max_paths: int = 5,
    base_dir: Optional[str] = None,
) -> str:
    if not trace_file.endswith(".json"):
        trace_file = f"{trace_file}.json"
    builder = DependencyTraceBuilder(
        profiler_buffer=profiler_buffer,
        task_types_to_str=task_types_to_str,
        wq_tensor=wq_tensor,
        num_tasks_tensor=num_tasks_tensor,
        scheduled_tasks=scheduled_tasks,
        base_dir=base_dir,
    )
    data = builder.build(max_paths=max_paths)
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return trace_file

