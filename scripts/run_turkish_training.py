"""Start a training stage and its TensorBoard mirror; same run_dir resumes."""

import argparse
import subprocess
import sys

from training.args import load_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=["smoke", "teacher", "student_smoke", "student", "student8_smoke", "student8"],
    )
    a = p.parse_args()
    config = f"training/configs/turkish_vast_{a.stage}.yaml"
    subprocess.run([sys.executable, "-m", "scripts.vast_preflight", config], check=True)
    args = load_args(config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    mirror = subprocess.Popen(
        [sys.executable, "-m", "scripts.turkish_tensorboard", str(args.run_dir), "--follow"]
    )
    try:
        code = subprocess.call([sys.executable, "-m", "training.train", config])
    finally:
        mirror.terminate()
        mirror.wait()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
