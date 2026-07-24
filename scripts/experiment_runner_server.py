#!/usr/bin/env python3
import json
import os
import shlex
import subprocess
import sys
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOST = "0.0.0.0"
PORT = 8000

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "ui_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {}
RUNS_LOCK = threading.Lock()
RUN_PROCESSES = {}


def load_dotenv_file(path):
    values = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export "):].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        values[key] = value

    return values


def resolve_env_defaults():
    candidates = [ROOT / ".env", ROOT / ".env-example"]

    for candidate in candidates:
        if candidate.exists():
            values = load_dotenv_file(candidate)
            return values, str(candidate)

    return {}, None


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def append_log(run_id, message):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if not run:
            return
        run["logs"].append({"at": now_iso(), "message": message})


def update_run(run_id, **updates):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if not run:
            return
        run.update(updates)


def read_stream(run_id, stream, prefix):
    try:
        for line in iter(stream.readline, ""):
            clean = line.rstrip("\n")
            if clean:
                append_log(run_id, f"{prefix}: {clean}")
    finally:
        stream.close()


def finalize_process(run_id, process):
    return_code = process.wait()
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if run and run.get("status") == "stopped":
            run["endedAt"] = now_iso()
            run["returnCode"] = return_code
        else:
            status = "completed" if return_code == 0 else "failed"
            if run:
                run["status"] = status
                run["endedAt"] = now_iso()
                run["returnCode"] = return_code
    append_log(run_id, f"Process exited with code {return_code}")
    with RUNS_LOCK:
        RUN_PROCESSES.pop(run_id, None)


def model_alias(value):
    if not value:
        return "gpt-4o-mini"

    aliases = {
        "gpt-4.1-mini": "gpt-4o-mini",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o"
    }
    return aliases.get(value, value)


def minutes_to_hhmmss(value):
    try:
        total_minutes = max(1, int(value))
    except (TypeError, ValueError):
        total_minutes = 1

    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}:00"


def choose_script(payload):
    # Default to navi script for IPA_* SUTs; safety script otherwise.
    sut = payload.get("executionParameters", {}).get("sut", "")
    if sut.startswith("IPA_"):
        return ROOT / "run_tests_navi.py"
    return ROOT / "run_tests_safety.py"


def build_command(run_id, payload):
    execution = payload.get("executionParameters", {})
    evaluation = payload.get("evaluationParameters", {}).get("fitnessFunctions", {})
    domain = payload.get("domainConfig", {})
    enable_wandb = bool(execution.get("enableWandb", False))

    script_path = choose_script(payload)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_config_path = run_dir / "features_config.json"
    feature_config_content = domain.get("featuresContent", "{}")
    feature_config_path.write_text(feature_config_content, encoding="utf-8")

    response_cfg = evaluation.get("response", {})
    content_cfg = evaluation.get("content", {})

    cmd = [
        sys.executable,
        str(script_path),
        "--sut",
        str(execution.get("sut", "IPA_LOS")),
        "--population_size",
        str(int(execution.get("populationSize", 2))),
        "--max_time",
        minutes_to_hhmmss(execution.get("testingTimeMinutes", 60)),
        "--th_answer",
        str(float(response_cfg.get("threshold", 0.75))),
        "--th_content",
        str(float(content_cfg.get("threshold", 0.75))),
        "--judge",
        model_alias(response_cfg.get("judgeModel", "gpt-4o-mini")),
        "--features_config",
        str(feature_config_path)
    ]

    # Single-turn maps to one generation in this bridge.
    turn_mode = execution.get("turnMode", "single")
    n_generations = 1 if turn_mode == "single" else 2
    cmd.extend(["--n_generations", str(n_generations)])

    # Avoid weave/wandb auth failure unless user explicitly enables logging.
    if not enable_wandb:
        cmd.append("--no_wandb")

    return cmd, run_dir


def build_process_env(payload):
    execution = payload.get("executionParameters", {})
    overrides = execution.get("envOverrides", {})
    enable_wandb = bool(execution.get("enableWandb", False))

    base_env = os.environ.copy()
    dotenv_values, _ = resolve_env_defaults()
    merged = {**base_env, **dotenv_values}

    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if not key:
                continue
            merged[str(key)] = "" if value is None else str(value)

    if enable_wandb:
        merged.setdefault("WANDB_ENTITY", "opentest")
        merged.setdefault("WANDB_PROJECT", "demo")

    return merged


def resolve_runtime_env():
    base_env = os.environ.copy()
    dotenv_values, source = resolve_env_defaults()
    merged = {**base_env, **dotenv_values}
    return merged, source


def serialize_run(run):
    return deepcopy(run)


class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _read_json(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        return json.loads(raw)

    def _write_json(self, payload, status=200):
        self._set_headers(status)
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        if path == "/health":
            self._write_json({"status": "ok", "time": now_iso()})
            return

        if path == "/experiments/runs":
            with RUNS_LOCK:
                runs = [serialize_run(run) for run in RUNS.values()]
            runs.sort(key=lambda item: item.get("startedAt", ""), reverse=True)
            self._write_json({"runs": runs})
            return

        if path in {"/experiments/env-defaults", "/env-defaults"}:
            defaults, source = resolve_env_defaults()
            self._write_json({
                "env": defaults,
                "source": source,
                "count": len(defaults)
            })
            return

        if path == "/wandb/runs":
            try:
                import wandb
            except Exception as exc:
                self._write_json({"error": f"wandb package not available: {exc}"}, status=501)
                return

            runtime_env, env_source = resolve_runtime_env()
            api_key = runtime_env.get("WANDB_API_KEY", "").strip()
            env_entity = runtime_env.get("WANDB_ENTITY", "").strip()
            env_project = runtime_env.get("WANDB_PROJECT", "").strip()

            entity = query.get("entity", [env_entity or "opentest"])[0]
            project = query.get("project", [env_project or "dev"])[0]
            limit_raw = query.get("limit", ["20"])[0]
            try:
                limit = max(1, min(100, int(limit_raw)))
            except ValueError:
                limit = 20

            if not api_key:
                self._write_json(
                    {
                        "error": "WANDB_API_KEY is missing. Define it in .env or .env-example.",
                        "envSource": env_source,
                    },
                    status=400,
                )
                return

            try:
                api = wandb.Api(api_key=api_key, timeout=20)
                api_runs = list(api.runs(f"{entity}/{project}", per_page=limit))
            except Exception as exc:
                self._write_json({"error": f"Failed to fetch W&B runs: {exc}"}, status=502)
                return

            runs = []
            for run in api_runs:
                summary = dict(run.summary or {})
                runs.append(
                    {
                        "id": run.id,
                        "name": run.name,
                        "state": run.state,
                        "url": run.url,
                        "createdAt": getattr(run, "created_at", None),
                        "updatedAt": getattr(run, "updated_at", None),
                        "project": project,
                        "entity": entity,
                        "summary": summary,
                    }
                )

            self._write_json(
                {
                    "entity": entity,
                    "project": project,
                    "count": len(runs),
                    "envSource": env_source,
                    "runs": runs,
                }
            )
            return

        if path.startswith("/experiments/runs/"):
            run_id = path.split("/")[-1]
            with RUNS_LOCK:
                run = RUNS.get(run_id)
            if not run:
                self._write_json({"error": f"Run '{run_id}' not found"}, status=404)
                return
            self._write_json(serialize_run(run))
            return

        self._write_json({"error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.startswith("/experiments/runs/") and path.endswith("/stop"):
            run_id = path.split("/")[-2]

            with RUNS_LOCK:
                run = RUNS.get(run_id)
                process = RUN_PROCESSES.get(run_id)

            if not run:
                self._write_json({"error": f"Run '{run_id}' not found"}, status=404)
                return

            if run.get("status") in {"completed", "failed", "stopped"}:
                self._write_json(
                    {
                        "message": f"Run already finished with status '{run.get('status')}'",
                        "runId": run_id,
                        "status": run.get("status")
                    },
                    status=200
                )
                return

            if process is None:
                update_run(run_id, status="stopped", endedAt=now_iso())
                append_log(run_id, "Stop requested, but no active process handle was found.")
                self._write_json(
                    {
                        "message": "Stop requested",
                        "runId": run_id,
                        "status": "stopped"
                    },
                    status=200
                )
                return

            append_log(run_id, "Stop requested from UI")
            try:
                process.terminate()
                update_run(run_id, status="stopped")
                self._write_json(
                    {
                        "message": "Stop signal sent",
                        "runId": run_id,
                        "status": "stopped"
                    },
                    status=200
                )
            except Exception as exc:
                append_log(run_id, f"Failed to stop process: {exc}")
                self._write_json({"error": str(exc), "runId": run_id}, status=500)

            return

        if path != "/experiments/start":
            self._write_json({"error": "Not found"}, status=404)
            return

        try:
            payload = self._read_json()
        except json.JSONDecodeError as exc:
            self._write_json({"error": f"Invalid JSON: {exc}"}, status=400)
            return

        run_id = f"run-{uuid.uuid4().hex[:10]}"

        try:
            cmd, run_dir = build_command(run_id, payload)
        except Exception as exc:
            self._write_json({"error": f"Failed to build command: {exc}"}, status=400)
            return

        run_record = {
            "id": run_id,
            "status": "running",
            "startedAt": now_iso(),
            "endedAt": None,
            "returnCode": None,
            "sut": payload.get("executionParameters", {}).get("sut", "IPA_LOS"),
            "command": " ".join(shlex.quote(item) for item in cmd),
            "runDir": str(run_dir),
            "logs": [
                {"at": now_iso(), "message": "Run accepted by backend"},
                {"at": now_iso(), "message": f"Command: {' '.join(shlex.quote(item) for item in cmd)}"}
            ]
        }

        with RUNS_LOCK:
            RUNS[run_id] = run_record

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=build_process_env(payload)
            )
            with RUNS_LOCK:
                RUN_PROCESSES[run_id] = process
        except Exception as exc:
            append_log(run_id, f"Failed to start process: {exc}")
            update_run(run_id, status="failed", endedAt=now_iso(), returnCode=-1)
            self._write_json({"error": str(exc), "runId": run_id}, status=500)
            return

        threading.Thread(target=read_stream, args=(run_id, process.stdout, "stdout"), daemon=True).start()
        threading.Thread(target=read_stream, args=(run_id, process.stderr, "stderr"), daemon=True).start()
        threading.Thread(target=finalize_process, args=(run_id, process), daemon=True).start()

        self._write_json(
            {
                "message": "Experiment process started",
                "runId": run_id,
                "status": "running"
            },
            status=202
        )


def main():
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"STELLAR runner server listening on http://{HOST}:{PORT}")
    print("POST /experiments/start")
    print("GET  /experiments/runs")
    print("GET  /experiments/runs/<run_id>")
    print("GET  /wandb/runs?entity=<entity>&project=<project>&limit=<n>")
    server.serve_forever()


if __name__ == "__main__":
    main()
