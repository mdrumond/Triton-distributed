"""Dependency trace exploration and visualization server.

This module provides a small utility that can ingest a dependency trace JSON
file exported by :mod:`dependency_trace` and expose a browser-based UI for
exploring the most critical execution paths.  The ingestion logic streams the
JSON file into a SQLite database so that even very large traces can be handled
without loading the entire payload into memory.

Example usage::

    python -m triton_dist.tools.profiler.critical_path_server \
        --trace /path/to/dependency_trace.json --serve

To inspect the top N critical paths directly from the command line::

    python -m triton_dist.tools.profiler.critical_path_server \
        --trace /path/to/dependency_trace.json --top 10

The script can also report the tasks that are common to a set of critical
paths::

    python -m triton_dist.tools.profiler.critical_path_server \
        --trace /path/to/dependency_trace.json --common 0 2 5

The ``--serve`` flag starts a lightweight HTTP server that serves both a JSON
API and a simple HTML UI for interactive exploration.  The UI allows choosing
how many critical paths to load, inspecting each path, and selecting multiple
paths to highlight tasks common to all selections.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse


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


def initialise_database(conn: sqlite3.Connection) -> None:
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
            origin TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS critical_paths (
            path_index INTEGER PRIMARY KEY,
            total_duration_ns INTEGER,
            slack_ns INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS critical_path_tasks (
            path_index INTEGER,
            ordinal INTEGER,
            task_node_id TEXT,
            PRIMARY KEY (path_index, ordinal)
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
    conn.execute("DELETE FROM critical_paths")
    conn.execute("DELETE FROM critical_path_tasks")
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
                            origin
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            obj.get("src"),
                            obj.get("dst"),
                            obj.get("start_tile"),
                            obj.get("end_tile"),
                            json.dumps(obj.get("origin")) if obj.get("origin") is not None else None,
                        ),
                    ),
                )
            elif key == "critical_paths":
                path_counter = 0

                def _insert_path(obj):
                    nonlocal path_counter
                    path_index = path_counter
                    path_counter += 1
                    conn.execute(
                        """
                        INSERT INTO critical_paths (
                            path_index,
                            total_duration_ns,
                            slack_ns
                        ) VALUES (?, ?, ?)
                        """,
                        (
                            path_index,
                            obj.get("total_duration_ns"),
                            obj.get("slack_ns"),
                        ),
                    )
                    tasks = obj.get("tasks", [])
                    for ordinal, node_id in enumerate(tasks):
                        conn.execute(
                            """
                            INSERT INTO critical_path_tasks (
                                path_index,
                                ordinal,
                                task_node_id
                            ) VALUES (?, ?, ?)
                            """,
                            (path_index, ordinal, node_id),
                        )

                _stream_array(fp, _insert_path)
            elif key == "min_start_time_ns":
                value = _read_scalar(fp)
                conn.execute(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    ("min_start_time_ns", json.dumps(value)),
                )
            else:
                _skip_value(fp)
            delim = _read_post_value_delimiter(fp)
            if delim == "}":
                break
            if delim not in {",", None}:
                raise ValueError(f"Unexpected delimiter {delim!r}")
    conn.commit()


class CriticalPathService:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @contextlib.contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def top_paths(self, limit: int) -> List[dict]:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT cp.path_index, cp.total_duration_ns, cp.slack_ns,
                       COUNT(cpt.task_node_id) AS length
                FROM critical_paths cp
                LEFT JOIN critical_path_tasks cpt
                    ON cp.path_index = cpt.path_index
                GROUP BY cp.path_index
                ORDER BY cp.total_duration_ns DESC, cp.path_index ASC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def path_tasks(self, path_index: int) -> List[dict]:
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT cpt.ordinal, t.*
                FROM critical_path_tasks cpt
                JOIN tasks t ON t.node_id = cpt.task_node_id
                WHERE cpt.path_index = ?
                ORDER BY cpt.ordinal ASC
                """,
                (path_index,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def common_tasks(self, path_indices: Iterable[int]) -> List[dict]:
        indices = list(path_indices)
        if not indices:
            return []
        placeholders = ",".join("?" for _ in indices)
        query = f"""
            SELECT t.*
            FROM tasks t
            JOIN (
                SELECT task_node_id
                FROM critical_path_tasks
                WHERE path_index IN ({placeholders})
                GROUP BY task_node_id
                HAVING COUNT(DISTINCT path_index) = ?
            ) shared ON shared.task_node_id = t.node_id
            ORDER BY t.start_time_ns ASC
        """
        with self._connection() as conn:
            cursor = conn.execute(query, (*indices, len(indices)))
            return [dict(row) for row in cursor.fetchall()]


HTML_PAGE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Critical Path Explorer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f7fa; }
    header { background: #1f2933; color: #fff; padding: 1rem 2rem; }
    main { padding: 1.5rem 2rem; }
    h1 { margin: 0; font-size: 1.75rem; }
    section { margin-bottom: 2rem; }
    table { border-collapse: collapse; width: 100%; background: #fff; }
    th, td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #e4e7eb; text-align: left; }
    th { background: #f0f4f8; }
    tr:hover { background: #f8fafc; }
    .path-row.selected { background: #e1effe; }
    .controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; }
    .badge { display: inline-block; background: #3b82f6; color: #fff; border-radius: 9999px; padding: 0.2rem 0.6rem; font-size: 0.8rem; }
    .panel { background: #fff; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08); }
    .task-list { list-style: none; padding: 0; margin: 0; max-height: 320px; overflow-y: auto; }
    .task-list li { padding: 0.5rem 0; border-bottom: 1px solid #e4e7eb; }
    .empty-state { color: #6b7280; font-style: italic; }
    .path-details { margin-top: 1rem; }
    button { padding: 0.4rem 0.9rem; border-radius: 0.4rem; border: none; background: #2563eb; color: #fff; cursor: pointer; }
    button:disabled { background: #9ca3af; cursor: not-allowed; }
    label { font-weight: 600; }
    input[type=number] { padding: 0.3rem 0.5rem; border-radius: 0.3rem; border: 1px solid #cbd5e1; width: 6rem; }
  </style>
</head>
<body>
  <header>
    <h1>Critical Path Explorer</h1>
  </header>
  <main>
    <section class=\"panel\">
      <div class=\"controls\">
        <label for=\"limit\">Top N paths:</label>
        <input id=\"limit\" type=\"number\" min=\"1\" max=\"200\" value=\"10\" />
        <button id=\"loadBtn\">Load</button>
      </div>
      <table>
        <thead>
          <tr>
            <th>Select</th>
            <th>#</th>
            <th>Total Duration (ms)</th>
            <th>Slack (ms)</th>
            <th>Tasks</th>
          </tr>
        </thead>
        <tbody id=\"pathsBody\"></tbody>
      </table>
    </section>
    <section class=\"panel\">
      <h2>Common Tasks</h2>
      <p id=\"commonSummary\" class=\"empty-state\">Select two or more paths to see overlapping tasks.</p>
      <ul id=\"commonTasks\" class=\"task-list\"></ul>
    </section>
    <section class=\"panel\">
      <h2>Path Details</h2>
      <div id=\"pathDetails\" class=\"empty-state\">Select a single path to view its tasks.</div>
    </section>
  </main>
  <script>
    const pathsBody = document.getElementById('pathsBody');
    const loadBtn = document.getElementById('loadBtn');
    const limitInput = document.getElementById('limit');
    const commonTasks = document.getElementById('commonTasks');
    const commonSummary = document.getElementById('commonSummary');
    const pathDetails = document.getElementById('pathDetails');

    let currentPaths = [];

    function ms(value) {
      if (value == null) return '-';
      return (value / 1e6).toFixed(3);
    }

    function renderPaths() {
      pathsBody.innerHTML = '';
      currentPaths.forEach((path, idx) => {
        const tr = document.createElement('tr');
        tr.classList.add('path-row');
        tr.dataset.index = path.path_index;
        tr.innerHTML = `
          <td><input type="checkbox" data-index="${path.path_index}" /></td>
          <td>${path.path_index}</td>
          <td>${ms(path.total_duration_ns)}</td>
          <td>${ms(path.slack_ns)}</td>
          <td><span class="badge">${path.length ?? 0}</span></td>
        `;
        const checkbox = tr.querySelector('input[type="checkbox"]');
        checkbox.addEventListener('change', () => {
          updateCommonTasks();
          highlightSelection();
        });
        tr.addEventListener('click', (event) => {
          if (event.target.tagName.toLowerCase() === 'input') {
            return;
          }
          pathsBody.querySelectorAll('tr').forEach(row => row.classList.remove('selected'));
          tr.classList.add('selected');
          loadPathDetails(path.path_index);
        });
        pathsBody.appendChild(tr);
      });
    }

    function selectedIndices() {
      return Array.from(pathsBody.querySelectorAll('input[type="checkbox"]'))
        .filter(cb => cb.checked)
        .map(cb => parseInt(cb.dataset.index, 10));
    }

    async function loadPaths() {
      const limit = parseInt(limitInput.value, 10) || 10;
      const response = await fetch(`/api/critical-paths?limit=${limit}`);
      currentPaths = await response.json();
      renderPaths();
      updateCommonTasks();
      pathDetails.innerHTML = 'Select a single path to view its tasks.';
      pathDetails.className = 'empty-state';
    }

    async function loadPathDetails(pathIndex) {
      const response = await fetch(`/api/critical-paths/${pathIndex}/tasks`);
      const tasks = await response.json();
      if (!tasks.length) {
        pathDetails.innerHTML = 'No tasks found for this path.';
        pathDetails.className = 'empty-state';
        return;
      }
      const list = document.createElement('ul');
      list.className = 'task-list';
      tasks.forEach(task => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${task.node_id}</strong> &mdash; ${task.task_type || 'unknown'} (Layer ${task.layer_id}, Task ${task.task_id}, Tile ${task.tile_id})<br />` +
          `SM ${task.sm_id}, start ${ms(task.start_time_ns)} ms, duration ${ms(task.duration_ns)} ms`;
        list.appendChild(li);
      });
      pathDetails.className = '';
      pathDetails.innerHTML = '';
      pathDetails.appendChild(list);
    }

    async function updateCommonTasks() {
      const indices = selectedIndices();
      if (indices.length < 2) {
        commonSummary.textContent = 'Select two or more paths to see overlapping tasks.';
        commonSummary.className = 'empty-state';
        commonTasks.innerHTML = '';
        return;
      }
      const response = await fetch('/api/common-tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path_indices: indices })
      });
      const tasks = await response.json();
      commonTasks.innerHTML = '';
      if (!tasks.length) {
        commonSummary.textContent = 'No common tasks between the selected paths.';
        commonSummary.className = 'empty-state';
        return;
      }
      commonSummary.textContent = `${tasks.length} common task(s) across paths ${indices.join(', ')}.`;
      commonSummary.className = '';
      tasks.forEach(task => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${task.node_id}</strong> &mdash; ${task.task_type || 'unknown'} (Layer ${task.layer_id}, Task ${task.task_id}, Tile ${task.tile_id})`;
        commonTasks.appendChild(li);
      });
    }

    function highlightSelection() {
      const indices = new Set(selectedIndices());
      pathsBody.querySelectorAll('tr').forEach(row => {
        if (indices.has(parseInt(row.dataset.index, 10))) {
          row.classList.add('selected');
        } else {
          row.classList.remove('selected');
        }
      });
    }

    loadBtn.addEventListener('click', () => {
      loadPaths();
    });

    loadPaths();
  </script>
</body>
</html>
"""


def make_handler(service: CriticalPathService):
    class _Handler(BaseHTTPRequestHandler):
        def _write_json(self, payload, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/critical-paths":
                query = parse_qs(parsed.query)
                limit = 10
                if "limit" in query:
                    try:
                        limit = max(1, min(500, int(query["limit"][0])))
                    except (ValueError, TypeError):
                        pass
                paths = service.top_paths(limit)
                self._write_json(paths)
                return
            if parsed.path.startswith("/api/critical-paths/"):
                try:
                    path_index = int(parsed.path.rsplit("/", 1)[-1])
                except ValueError:
                    self._write_json({"error": "invalid path index"}, status=400)
                    return
                tasks = service.path_tasks(path_index)
                self._write_json(tasks)
                return
            self._write_json({"error": "not found"}, status=404)

        def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            parsed = urlparse(self.path)
            if parsed.path != "/api/common-tasks":
                self._write_json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            try:
                body = json.loads(payload or b"{}")
            except json.JSONDecodeError:
                self._write_json({"error": "invalid json"}, status=400)
                return
            indices = body.get("path_indices", [])
            try:
                normalized = [int(x) for x in indices]
            except (TypeError, ValueError):
                self._write_json({"error": "path_indices must be integers"}, status=400)
                return
            tasks = service.common_tasks(normalized)
            self._write_json(tasks)

        def log_message(self, format, *args):  # noqa: A003 - follow BaseHTTPRequestHandler signature
            sys.stderr.write("[critical_path_server] " + (format % args) + "\n")

    return _Handler


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore dependency trace critical paths")
    parser.add_argument("--trace", required=True, help="Path to dependency trace JSON file")
    parser.add_argument(
        "--database",
        help="Optional path to a SQLite database for caching the ingested trace",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Print the top N critical paths to stdout and exit",
    )
    parser.add_argument(
        "--common",
        nargs="*",
        help="Print tasks common to the specified critical path indices and exit",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start an HTTP server with a browser UI after ingesting the trace",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address for the HTTP server")
    parser.add_argument("--port", type=int, default=8765, help="Port for the HTTP server")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise SystemExit(f"Trace file {trace_path} does not exist")
    if args.database:
        db_path = Path(args.database)
    else:
        db_fd, db_name = tempfile.mkstemp(prefix="critical_paths_", suffix=".sqlite3")
        os.close(db_fd)
        db_path = Path(db_name)
    conn = sqlite3.connect(str(db_path))
    try:
        ingest_trace(trace_path, conn)
    finally:
        conn.close()

    service = CriticalPathService(db_path)

    if args.top:
        top_paths = service.top_paths(args.top)
        for row in top_paths:
            print(
                f"Path {row['path_index']}: duration={row['total_duration_ns']} ns, "
                f"slack={row['slack_ns']} ns, tasks={row['length']}"
            )
    if args.common:
        try:
            indices = [int(idx) for idx in args.common]
        except ValueError:
            raise SystemExit("All --common values must be integers")
        tasks = service.common_tasks(indices)
        if not tasks:
            print("No common tasks across the specified critical paths.")
        else:
            print(f"Common tasks for paths {', '.join(map(str, indices))}:")
            for task in tasks:
                print(
                    f"  {task['node_id']} type={task['task_type']} layer={task['layer_id']} "
                    f"task={task['task_id']} tile={task['tile_id']}"
                )
    if not args.serve:
        if not args.database:
            with contextlib.suppress(FileNotFoundError):
                os.remove(db_path)
        return

    handler_cls = make_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Serving critical path explorer on http://{args.host}:{args.port}")
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

