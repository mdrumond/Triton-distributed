"""Dependency trace inspection server.

This module ingests the dependency trace JSON produced by
``dependency_trace.export_dependency_trace`` and exposes a lightweight HTTP
interface for inspecting individual tasks.  Users can browse the tasks,
select one, and view both its producers and consumers together with the code
locations that established the dependencies.  The ingestion path streams the
JSON payload into a SQLite database so large traces can be explored without
loading them fully into memory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import linecache
import os
import sqlite3
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


# --------------------------------------------------------------------------------------
# Streaming JSON helpers
# --------------------------------------------------------------------------------------

def _read_next_non_ws(fp) -> str:
    while True:
        ch = fp.read(1)
        if not ch:
            return ""
        if ch not in " \t\r\n":
            return ch


def _expect_char(fp, expected: str) -> None:
    ch = _read_next_non_ws(fp)
    if ch != expected:
        raise ValueError(f"Expected character {expected!r}, got {ch!r}")


def _skip_whitespace(fp) -> None:
    while True:
        pos = fp.tell()
        ch = fp.read(1)
        if not ch:
            return
        if ch in " \t\r\n":
            continue
        fp.seek(pos)
        return


def _read_key(fp) -> Optional[str]:
    ch = _read_next_non_ws(fp)
    if ch == "":
        return None
    if ch == "}":
        return None
    if ch != '"':
        raise ValueError(f"Expected start of string for object key, got {ch!r}")
    key_chars: List[str] = []
    escape = False
    while True:
        nxt = fp.read(1)
        if nxt == "":
            raise ValueError("Unexpected EOF while reading object key")
        if escape:
            key_chars.append(nxt)
        elif nxt == "\\":
            escape = True
            key_chars.append(nxt)
            continue
        elif nxt == '"':
            break
        else:
            key_chars.append(nxt)
        escape = False
    key = json.loads('"' + ''.join(key_chars) + '"')
    ch = _read_next_non_ws(fp)
    if ch != ":":
        raise ValueError(f"Expected ':' after key {key!r}, got {ch!r}")
    return key


def _read_json_object(fp, first_char: str) -> str:
    if first_char != "{":
        raise ValueError("JSON object must start with '{'")
    buf = [first_char]
    depth = 1
    in_string = False
    escape = False
    while depth > 0:
        ch = fp.read(1)
        if ch == "":
            raise ValueError("Unexpected EOF while parsing JSON object")
        buf.append(ch)
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    return "".join(buf)


def _stream_array(fp, callback: Callable[[dict], None]) -> None:
    _expect_char(fp, "[")
    while True:
        ch = _read_next_non_ws(fp)
        if ch == "":
            raise ValueError("Unexpected EOF while reading array")
        if ch == "]":
            return
        if ch != "{":
            raise ValueError(f"Expected '{{' to start array element, got {ch!r}")
        obj_str = _read_json_object(fp, ch)
        callback(json.loads(obj_str))
        ch = _read_next_non_ws(fp)
        if ch == "]":
            return
        if ch != ",":
            raise ValueError(f"Expected ',' between array elements, got {ch!r}")


def _read_scalar(fp):
    _skip_whitespace(fp)
    decoder = json.JSONDecoder()
    start_pos = fp.tell()
    buffer = ""
    while True:
        chunk = fp.read(64)
        if chunk == "":
            raise ValueError("Unexpected EOF while reading scalar value")
        buffer += chunk
        try:
            value, index = decoder.raw_decode(buffer)
            fp.seek(start_pos + index)
            return value
        except json.JSONDecodeError:
            continue


def _skip_value(fp) -> None:
    _skip_whitespace(fp)
    decoder = json.JSONDecoder()
    start_pos = fp.tell()
    buffer = ""
    while True:
        chunk = fp.read(64)
        if chunk == "":
            raise ValueError("Unexpected EOF while skipping value")
        buffer += chunk
        try:
            _, index = decoder.raw_decode(buffer)
            fp.seek(start_pos + index)
            return
        except json.JSONDecodeError:
            continue


def _read_post_value_delimiter(fp) -> Optional[str]:
    while True:
        ch = fp.read(1)
        if ch == "":
            return None
        if ch in " \t\r\n":
            continue
        if ch in {",", "}"}:
            return ch
        raise ValueError(f"Unexpected character {ch!r} after JSON value")


# --------------------------------------------------------------------------------------
# Database ingestion
# --------------------------------------------------------------------------------------


def initialise_database(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS critical_paths")
    conn.execute("DROP TABLE IF EXISTS critical_path_tasks")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            node_id TEXT PRIMARY KEY,
            task_type TEXT,
            task_type_id INTEGER,
            layer_id INTEGER,
            task_id INTEGER,
            tile_id INTEGER,
            sm_id INTEGER,
            start_time_ns INTEGER,
            duration_ns INTEGER,
            finish_time_ns INTEGER,
            absolute_start_time_ns INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dependencies (
            src TEXT,
            dst TEXT,
            start_tile INTEGER,
            end_tile INTEGER,
            origin_filename TEXT,
            origin_lineno INTEGER,
            origin_function TEXT,
            origin_code_context TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()


def clear_database(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM tasks")
    conn.execute("DELETE FROM dependencies")
    conn.execute("DELETE FROM metadata")
    conn.commit()


def ingest_trace(trace_path: Path, conn: sqlite3.Connection) -> None:
    initialise_database(conn)
    clear_database(conn)
    conn.execute("BEGIN")
    with trace_path.open("r", encoding="utf-8") as fp:
        _expect_char(fp, "{")
        while True:
            key = _read_key(fp)
            if key is None:
                break
            if key == "tasks":
                _stream_array(
                    fp,
                    lambda obj: conn.execute(
                        """
                        INSERT INTO tasks (
                            node_id,
                            task_type,
                            task_type_id,
                            layer_id,
                            task_id,
                            tile_id,
                            sm_id,
                            start_time_ns,
                            duration_ns,
                            finish_time_ns,
                            absolute_start_time_ns
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            obj.get("node_id"),
                            obj.get("task_type"),
                            obj.get("task_type_id"),
                            obj.get("layer_id"),
                            obj.get("task_id"),
                            obj.get("tile_id"),
                            obj.get("sm_id"),
                            obj.get("start_time_ns"),
                            obj.get("duration_ns"),
                            obj.get("finish_time_ns"),
                            obj.get("absolute_start_time_ns"),
                        ),
                    ),
                )
            elif key == "dependencies":
                _stream_array(
                    fp,
                    lambda obj: conn.execute(
                        """
                        INSERT INTO dependencies (
                            src,
                            dst,
                            start_tile,
                            end_tile,
                            origin_filename,
                            origin_lineno,
                            origin_function,
                            origin_code_context
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            obj.get("src"),
                            obj.get("dst"),
                            obj.get("start_tile"),
                            obj.get("end_tile"),
                            (obj.get("origin") or {}).get("filename"),
                            (obj.get("origin") or {}).get("lineno"),
                            (obj.get("origin") or {}).get("function"),
                            (obj.get("origin") or {}).get("code_context"),
                        ),
                    ),
                )
            elif key == "min_start_time_ns":
                value = _read_scalar(fp)
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    ("min_start_time_ns", json.dumps(value)),
                )
            elif key == "origin_base_dir":
                value = _read_scalar(fp)
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    ("origin_base_dir", json.dumps(value)),
                )
            else:
                _skip_value(fp)
            delim = _read_post_value_delimiter(fp)
            if delim == "}":
                break
            if delim not in {",", None}:
                raise ValueError(f"Unexpected delimiter {delim!r}")
    conn.commit()


# --------------------------------------------------------------------------------------
# Query service
# --------------------------------------------------------------------------------------


class DependencyTraceService:
    def __init__(self, db_path: Path, trace_path: Path) -> None:
        self._db_path = db_path
        self._trace_path = trace_path
        self._metadata_cache: Optional[Dict[str, Optional[str]]] = None

    @contextlib.contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _load_metadata(self) -> Dict[str, Optional[str]]:
        if self._metadata_cache is not None:
            return self._metadata_cache
        data: Dict[str, Optional[str]] = {}
        with self._connection() as conn:
            cursor = conn.execute("SELECT key, value FROM metadata")
            for row in cursor.fetchall():
                value = row["value"]
                try:
                    data[row["key"]] = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    data[row["key"]] = value
        self._metadata_cache = data
        return data

    @property
    def metadata(self) -> Dict[str, Optional[str]]:
        return self._load_metadata()

    def _resolve_source_path(self, filename: Optional[str]) -> Optional[str]:
        if not filename:
            return None
        if os.path.isabs(filename) and os.path.exists(filename):
            return os.path.abspath(filename)
        candidates: List[str] = []
        base_dir = self.metadata.get("origin_base_dir")
        if base_dir:
            candidates.append(os.path.join(base_dir, filename))
        candidates.append(os.path.join(self._trace_path.parent, filename))
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)
        return None

    def _source_info(self, filename: Optional[str], lineno: Optional[int], function: Optional[str], code_context: Optional[str]) -> Dict[str, Optional[str]]:
        resolved = self._resolve_source_path(filename)
        error: Optional[str] = None
        source_line: Optional[str] = None
        if resolved and lineno:
            line = linecache.getline(resolved, int(lineno))
            if line:
                source_line = line.rstrip("\n")
            else:
                error = "Unable to read source line"
        elif filename and lineno:
            error = "Source file not found"
        return {
            "requested_path": filename,
            "resolved_path": resolved,
            "lineno": lineno,
            "function": function,
            "code_context": code_context,
            "source_line": source_line,
            "error": error,
        }

    def list_tasks(self, limit: int, offset: int, query: Optional[str]) -> Dict[str, object]:
        search = (query or "").strip().lower()
        where_clause = ""
        params: List[object] = []
        if search:
            pattern = f"%{search}%"
            where_clause = "WHERE LOWER(node_id) LIKE ? OR LOWER(task_type) LIKE ? OR CAST(layer_id AS TEXT) LIKE ?"
            params.extend([pattern, pattern, pattern])
        with self._connection() as conn:
            total_cursor = conn.execute(f"SELECT COUNT(*) FROM tasks {where_clause}", params)
            total = int(total_cursor.fetchone()[0])
            query_sql = (
                "SELECT node_id, task_type, task_type_id, layer_id, task_id, tile_id, sm_id, "
                "start_time_ns, duration_ns, finish_time_ns, absolute_start_time_ns "
                f"FROM tasks {where_clause} ORDER BY start_time_ns ASC LIMIT ? OFFSET ?"
            )
            rows = conn.execute(query_sql, (*params, limit, offset)).fetchall()
            tasks = [dict(row) for row in rows]
        return {"tasks": tasks, "total": total, "offset": offset, "limit": limit}

    def _row_to_task(self, row: sqlite3.Row, prefix: str) -> Dict[str, object]:
        return {
            "node_id": row[f"{prefix}_node_id"],
            "task_type": row[f"{prefix}_task_type"],
            "task_type_id": row[f"{prefix}_task_type_id"],
            "layer_id": row[f"{prefix}_layer_id"],
            "task_id": row[f"{prefix}_task_id"],
            "tile_id": row[f"{prefix}_tile_id"],
            "sm_id": row[f"{prefix}_sm_id"],
            "start_time_ns": row[f"{prefix}_start_time_ns"],
            "duration_ns": row[f"{prefix}_duration_ns"],
            "finish_time_ns": row[f"{prefix}_finish_time_ns"],
            "absolute_start_time_ns": row[f"{prefix}_absolute_start_time_ns"],
        }

    def _collect_dependencies(self, conn: sqlite3.Connection, node_id: str, incoming: bool) -> List[Dict[str, object]]:
        if incoming:
            query = (
                "SELECT d.src, d.dst, d.start_tile, d.end_tile, d.origin_filename, d.origin_lineno, "
                "d.origin_function, d.origin_code_context, "
                "t.node_id AS producer_node_id, t.task_type AS producer_task_type, "
                "t.task_type_id AS producer_task_type_id, t.layer_id AS producer_layer_id, "
                "t.task_id AS producer_task_id, t.tile_id AS producer_tile_id, t.sm_id AS producer_sm_id, "
                "t.start_time_ns AS producer_start_time_ns, t.duration_ns AS producer_duration_ns, "
                "t.finish_time_ns AS producer_finish_time_ns, t.absolute_start_time_ns AS producer_absolute_start_time_ns "
                "FROM dependencies d "
                "JOIN tasks t ON t.node_id = d.src "
                "WHERE d.dst = ? "
                "ORDER BY t.start_time_ns ASC"
            )
            role_key = "producer"
        else:
            query = (
                "SELECT d.src, d.dst, d.start_tile, d.end_tile, d.origin_filename, d.origin_lineno, "
                "d.origin_function, d.origin_code_context, "
                "t.node_id AS consumer_node_id, t.task_type AS consumer_task_type, "
                "t.task_type_id AS consumer_task_type_id, t.layer_id AS consumer_layer_id, "
                "t.task_id AS consumer_task_id, t.tile_id AS consumer_tile_id, t.sm_id AS consumer_sm_id, "
                "t.start_time_ns AS consumer_start_time_ns, t.duration_ns AS consumer_duration_ns, "
                "t.finish_time_ns AS consumer_finish_time_ns, t.absolute_start_time_ns AS consumer_absolute_start_time_ns "
                "FROM dependencies d "
                "JOIN tasks t ON t.node_id = d.dst "
                "WHERE d.src = ? "
                "ORDER BY t.start_time_ns ASC"
            )
            role_key = "consumer"
        rows = conn.execute(query, (node_id,)).fetchall()
        results: List[Dict[str, object]] = []
        for row in rows:
            dependency = {
                "src": row["src"],
                "dst": row["dst"],
                "start_tile": row["start_tile"],
                "end_tile": row["end_tile"],
                "origin": {
                    "filename": row["origin_filename"],
                    "lineno": row["origin_lineno"],
                    "function": row["origin_function"],
                    "code_context": row["origin_code_context"],
                },
            }
            if incoming:
                task_info = self._row_to_task(row, "producer")
            else:
                task_info = self._row_to_task(row, "consumer")
            source = self._source_info(
                dependency["origin"].get("filename"),
                dependency["origin"].get("lineno"),
                dependency["origin"].get("function"),
                dependency["origin"].get("code_context"),
            )
            results.append({
                "dependency": dependency,
                role_key: task_info,
                "source": source,
            })
        return results

    def task_detail(self, node_id: str) -> Optional[Dict[str, object]]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT node_id, task_type, task_type_id, layer_id, task_id, tile_id, sm_id, "
                "start_time_ns, duration_ns, finish_time_ns, absolute_start_time_ns FROM tasks WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            if row is None:
                return None
            task = dict(row)
            inputs = self._collect_dependencies(conn, node_id, incoming=True)
            dependents = self._collect_dependencies(conn, node_id, incoming=False)
        return {"task": task, "inputs": inputs, "dependents": dependents}


# --------------------------------------------------------------------------------------
# HTTP API and UI
# --------------------------------------------------------------------------------------


HTML_PAGE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Dependency Trace Inspector</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; margin: 0; background: #f6f7fb; color: #1f2933; }
    header { background: #111827; color: #f9fafb; padding: 1.2rem 2rem; }
    header h1 { margin: 0; font-size: 1.75rem; }
    main { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; padding: 1.5rem 2rem 2.5rem; }
    .panel { background: #ffffff; border-radius: 0.75rem; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); padding: 1.25rem; display: flex; flex-direction: column; }
    .panel h2 { margin-top: 0; font-size: 1.25rem; }
    .controls { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1rem; align-items: center; }
    label { font-weight: 600; }
    input[type=text], input[type=number] { padding: 0.4rem 0.6rem; border-radius: 0.5rem; border: 1px solid #cbd5e1; min-width: 10rem; }
    button { padding: 0.45rem 0.9rem; border-radius: 0.5rem; border: none; background: #2563eb; color: #fff; cursor: pointer; font-weight: 600; }
    button:disabled { background: #9ca3af; cursor: not-allowed; }
    table { width: 100%; border-collapse: collapse; border-radius: 0.5rem; overflow: hidden; box-shadow: inset 0 0 0 1px #e2e8f0; }
    thead { background: #f1f5f9; }
    th, td { padding: 0.55rem 0.75rem; text-align: left; font-size: 0.9rem; border-bottom: 1px solid #e2e8f0; }
    tr:hover { background: #eef2ff; }
    tr.selected { background: #dbeafe; }
    .empty-state { color: #6b7280; font-style: italic; }
    .dependency-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.75rem; }
    .dependency-item { border: 1px solid #e2e8f0; border-radius: 0.65rem; padding: 0.75rem 0.9rem; background: #f8fafc; }
    .dependency-item h4 { margin: 0 0 0.35rem 0; font-size: 1rem; color: #1d4ed8; }
    .dependency-item pre { margin: 0.5rem 0 0 0; padding: 0.6rem; background: #0f172a; color: #f1f5f9; border-radius: 0.5rem; overflow-x: auto; }
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr)); gap: 0.6rem; margin-bottom: 1rem; }
    .meta-card { background: #f8fafc; border-radius: 0.6rem; padding: 0.6rem 0.75rem; border: 1px solid #e2e8f0; }
    .meta-card span { display: block; font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem; }
    @media (max-width: 1200px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Dependency Trace Inspector</h1>
  </header>
  <main>
    <section class=\"panel\">
      <h2>Tasks</h2>
      <div class=\"controls\">
        <label for=\"search\">Search</label>
        <input id=\"search\" type=\"text\" placeholder=\"Node id, task type, or layer\" />
        <label for=\"limit\">Limit</label>
        <input id=\"limit\" type=\"number\" min=\"1\" max=\"500\" value=\"50\" />
        <button id=\"loadBtn\">Load</button>
        <button id=\"prevBtn\">Prev</button>
        <button id=\"nextBtn\">Next</button>
      </div>
      <table>
        <thead>
          <tr>
            <th>Node</th>
            <th>Type</th>
            <th>Layer</th>
            <th>Task</th>
            <th>Tile</th>
            <th>Start (ms)</th>
            <th>Duration (µs)</th>
          </tr>
        </thead>
        <tbody id=\"tasksBody\"></tbody>
      </table>
      <p id=\"tasksStatus\" class=\"empty-state\" style=\"margin-top:0.75rem;\"></p>
    </section>
    <section class=\"panel\">
      <h2>Task Details</h2>
      <div id=\"taskDetails\" class=\"empty-state\">Select a task to inspect its dependencies.</div>
    </section>
  </main>
  <script>
    const tasksBody = document.getElementById('tasksBody');
    const tasksStatus = document.getElementById('tasksStatus');
    const taskDetails = document.getElementById('taskDetails');
    const searchInput = document.getElementById('search');
    const limitInput = document.getElementById('limit');
    const loadBtn = document.getElementById('loadBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');

    let offset = 0;
    let total = 0;
    let currentSelection = null;

    function formatNs(ns) {
      if (typeof ns !== 'number') return '—';
      return (ns / 1e6).toFixed(3);
    }

    function formatUs(ns) {
      if (typeof ns !== 'number') return '—';
      return (ns / 1e3).toFixed(2);
    }

    function updatePager() {
      prevBtn.disabled = offset <= 0;
      const limit = parseInt(limitInput.value, 10) || 50;
      nextBtn.disabled = offset + limit >= total;
      const end = Math.min(total, offset + limit);
      if (total === 0) {
        tasksStatus.textContent = 'No tasks found for the current filter.';
      } else {
        tasksStatus.textContent = `Showing ${offset + 1}–${end} of ${total} tasks.`;
      }
    }

    function clearTasks() {
      tasksBody.innerHTML = '';
    }

    function renderTasks(tasks) {
      clearTasks();
      tasks.forEach(task => {
        const row = document.createElement('tr');
        row.dataset.nodeId = task.node_id;
        if (task.node_id === currentSelection) {
          row.classList.add('selected');
        }
        const cells = [
          task.node_id,
          task.task_type,
          task.layer_id,
          task.task_id,
          task.tile_id,
          formatNs(task.start_time_ns),
          formatUs(task.duration_ns),
        ];
        cells.forEach(text => {
          const cell = document.createElement('td');
          cell.textContent = text ?? '—';
          row.appendChild(cell);
        });
        row.addEventListener('click', () => {
          currentSelection = task.node_id;
          document.querySelectorAll('tr.selected').forEach(el => el.classList.remove('selected'));
          row.classList.add('selected');
          loadTaskDetails(task.node_id);
        });
        tasksBody.appendChild(row);
      });
      updatePager();
    }

    function renderDependencySection(container, title, items, roleLabel) {
      const heading = document.createElement('h3');
      heading.textContent = title;
      container.appendChild(heading);
      if (!items.length) {
        const empty = document.createElement('p');
        empty.className = 'empty-state';
        empty.textContent = 'None';
        container.appendChild(empty);
        return;
      }
      const list = document.createElement('ul');
      list.className = 'dependency-list';
      items.forEach(item => {
        const entry = document.createElement('li');
        entry.className = 'dependency-item';
        const header = document.createElement('h4');
        const role = item[roleLabel];
        header.textContent = `${role.node_id} • layer ${role.layer_id} task ${role.task_id} tile ${role.tile_id}`;
        entry.appendChild(header);

        const meta = document.createElement('div');
        meta.textContent = `Tiles ${item.dependency.start_tile} – ${item.dependency.end_tile}`;
        entry.appendChild(meta);

        const origin = item.source;
        const originInfo = document.createElement('div');
        let originText = '';
        if (origin.function) {
          originText += `${origin.function} `;
        }
        if (origin.requested_path) {
          originText += origin.requested_path;
        }
        if (origin.lineno) {
          originText += `:${origin.lineno}`;
        }
        if (!originText) {
          originText = 'Origin unknown';
        }
        if (origin.error) {
          originText += ` — ${origin.error}`;
        }
        originInfo.textContent = originText;
        entry.appendChild(originInfo);

        if (origin.source_line) {
          const pre = document.createElement('pre');
          pre.textContent = origin.source_line;
          entry.appendChild(pre);
        } else if (origin.code_context) {
          const pre = document.createElement('pre');
          pre.textContent = origin.code_context;
          entry.appendChild(pre);
        }
        list.appendChild(entry);
      });
      container.appendChild(list);
    }

    function renderTaskDetails(data) {
      taskDetails.innerHTML = '';
      if (!data || !data.task) {
        taskDetails.className = 'empty-state';
        taskDetails.textContent = 'Task not found.';
        return;
      }
      taskDetails.className = '';

      const title = document.createElement('h3');
      title.textContent = data.task.node_id;
      taskDetails.appendChild(title);

      const metaGrid = document.createElement('div');
      metaGrid.className = 'meta-grid';
      const entries = [
        ['Task type', data.task.task_type],
        ['Layer', data.task.layer_id],
        ['Task id', data.task.task_id],
        ['Tile', data.task.tile_id],
        ['Start (ns)', data.task.start_time_ns],
        ['Duration (ns)', data.task.duration_ns],
        ['Finish (ns)', data.task.finish_time_ns],
        ['SM id', data.task.sm_id],
      ];
      entries.forEach(([label, value]) => {
        const card = document.createElement('div');
        card.className = 'meta-card';
        const span = document.createElement('span');
        span.textContent = label;
        const strong = document.createElement('strong');
        strong.textContent = value ?? '—';
        card.appendChild(span);
        card.appendChild(strong);
        metaGrid.appendChild(card);
      });
      taskDetails.appendChild(metaGrid);

      renderDependencySection(taskDetails, 'Input dependencies', data.inputs || [], 'producer');
      renderDependencySection(taskDetails, 'Dependents', data.dependents || [], 'consumer');
    }

    async function loadTasks() {
      const limit = parseInt(limitInput.value, 10) || 50;
      const params = new URLSearchParams();
      params.set('limit', limit);
      params.set('offset', offset);
      const search = searchInput.value.trim();
      if (search) {
        params.set('q', search);
      }
      const response = await fetch(`/api/tasks?${params.toString()}`);
      if (!response.ok) {
        tasksStatus.textContent = 'Failed to load tasks.';
        return;
      }
      const data = await response.json();
      total = data.total;
      offset = data.offset;
      renderTasks(data.tasks);
    }

    async function loadTaskDetails(nodeId) {
      const response = await fetch(`/api/task/${encodeURIComponent(nodeId)}`);
      if (!response.ok) {
        taskDetails.className = 'empty-state';
        taskDetails.textContent = 'Failed to load task details.';
        return;
      }
      const data = await response.json();
      renderTaskDetails(data);
    }

    loadBtn.addEventListener('click', () => {
      offset = 0;
      loadTasks();
    });
    prevBtn.addEventListener('click', () => {
      const limit = parseInt(limitInput.value, 10) || 50;
      offset = Math.max(0, offset - limit);
      loadTasks();
    });
    nextBtn.addEventListener('click', () => {
      const limit = parseInt(limitInput.value, 10) || 50;
      offset = offset + limit;
      loadTasks();
    });

    window.addEventListener('DOMContentLoaded', () => {
      loadTasks();
    });
  </script>
</body>
</html>
"""


def make_handler(service: DependencyTraceService):
    class _Handler(BaseHTTPRequestHandler):
        def _write_json(self, payload: Dict[str, object], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/tasks":
                qs = parse_qs(parsed.query or "")
                try:
                    limit = int(qs.get("limit", ["50"])[0])
                except ValueError:
                    self._write_json({"error": "limit must be an integer"}, status=400)
                    return
                try:
                    offset = int(qs.get("offset", ["0"])[0])
                except ValueError:
                    self._write_json({"error": "offset must be an integer"}, status=400)
                    return
                limit = max(1, min(limit, 500))
                offset = max(0, offset)
                query = qs.get("q", [""])[0]
                payload = service.list_tasks(limit=limit, offset=offset, query=query)
                self._write_json(payload)
                return
            if parsed.path.startswith("/api/task/"):
                node_id = parsed.path[len("/api/task/") :]
                try:
                    node_id = parse_qs(f"node={node_id}")["node"][0]
                except Exception:  # noqa: BLE001 - defensive parsing
                    pass
                detail = service.task_detail(node_id)
                if detail is None:
                    self._write_json({"error": "task not found"}, status=404)
                else:
                    self._write_json(detail)
                return
            self._write_json({"error": "not found"}, status=404)

        def log_message(self, format, *args):  # noqa: A003
            sys.stderr.write("[dependency_trace_server] " + (format % args) + "\n")

    return _Handler


# --------------------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an inspector for dependency traces")
    parser.add_argument("--trace", required=True, help="Path to dependency trace JSON file")
    parser.add_argument(
        "--database",
        help="Optional path to a SQLite database used to cache the ingested trace",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address for the HTTP server")
    parser.add_argument("--port", type=int, default=8765, help="Port number for the HTTP server")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise SystemExit(f"Trace file {trace_path} does not exist")
    if args.database:
        db_path = Path(args.database)
    else:
        db_fd, db_name = tempfile.mkstemp(prefix="dependency_trace_", suffix=".sqlite3")
        os.close(db_fd)
        db_path = Path(db_name)
    conn = sqlite3.connect(str(db_path))
    try:
        ingest_trace(trace_path, conn)
    finally:
        conn.close()

    service = DependencyTraceService(db_path, trace_path)
    handler_cls = make_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Serving dependency trace inspector on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server.server_close()
        if not args.database:
            with contextlib.suppress(FileNotFoundError):
                os.remove(db_path)


if __name__ == "__main__":
    main()
