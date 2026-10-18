"""Register a resumable teacher or student training stage under Supervisor."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["teacher", "student_smoke", "student", "student8_smoke", "student8"]
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    name = f"pocket-turkish-{args.stage}"
    config = Path(f"/etc/supervisor/conf.d/{name}.conf")
    if config.exists():
        raise RuntimeError(f"{name} already exists; inspect its status before restarting")
    log = repo / f"runs/turkish_vast_{args.stage}/supervisor.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f"""[program:{name}]
command=/venv/main/bin/python -u -m scripts.run_turkish_training {args.stage}
directory={repo}
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={log}
stdout_logfile_maxbytes=20000000
stdout_logfile_backups=2
environment=PYTHONUNBUFFERED="1",PYTHONUTF8="1"
""",
        encoding="utf-8",
    )
    for command in (("reread",), ("update",), ("start", name)):
        subprocess.run(["supervisorctl", *command], check=True)


if __name__ == "__main__":
    main()
