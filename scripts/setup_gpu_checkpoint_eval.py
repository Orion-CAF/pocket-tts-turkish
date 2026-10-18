"""Run held-out Turkish checkpoint evaluation under Vast's Supervisor.

Example:
    python -m scripts.setup_gpu_checkpoint_eval teacher 2000 --limit 20
"""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["teacher", "student", "student8"])
    parser.add_argument("step", type=int)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--checkpoint", type=Path, help="Preserved checkpoint path")
    args = parser.parse_args()
    if args.step < 1 or args.limit < 1:
        parser.error("step and limit must be positive")

    repo = Path(__file__).resolve().parents[1]
    run = repo / f"runs/turkish_vast_{args.stage}"
    checkpoint = args.checkpoint or run / f"checkpoint_{args.step:08d}.pt"
    if not checkpoint.is_file():
        parser.error(f"checkpoint missing: {checkpoint}")
    manifest = repo / f"data/tr/v1/{args.split}_aligned.jsonl"
    if not manifest.is_file():
        parser.error(f"manifest missing: {manifest}")
    output = run / f"eval_{args.split}_{args.step:08d}_{args.limit}"
    if output.exists():
        parser.error(f"evaluation output already exists: {output}")

    name = f"pocket-turkish-eval-{args.stage}-{args.split}-{args.step}-{args.limit}"
    config = Path(f"/etc/supervisor/conf.d/{name}.conf")
    if config.exists():
        parser.error(f"Supervisor job already exists: {name}")
    log = run / f"eval_{args.split}_{args.step:08d}_{args.limit}.log"
    command = (
        f"/venv/main/bin/python -u -m scripts.evaluate_turkish {run} "
        f"--manifest {manifest} --checkpoint {checkpoint} --output {output} "
        f"--cfg {2.0 if args.stage == 'teacher' else 1.0} --limit {args.limit}"
    )
    config.write_text(
        f"""[program:{name}]
command={command}
directory={repo}
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={log}
stdout_logfile_maxbytes=20000000
stdout_logfile_backups=1
environment=PYTHONUNBUFFERED="1",PYTHONUTF8="1"
""",
        encoding="utf-8",
    )
    for command_args in (("reread",), ("update",), ("start", name)):
        subprocess.run(["supervisorctl", *command_args], check=True)
    print(f"evaluation job={name} log={log} output={output}")


if __name__ == "__main__":
    main()
