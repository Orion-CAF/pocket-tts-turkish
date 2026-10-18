"""Mirror upstream progress.jsonl scalars to TensorBoard without changing training."""

import argparse
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--follow", action="store_true")
    a = p.parse_args()
    offset = 0
    with SummaryWriter(str(a.run / "tensorboard")) as writer:
        while True:
            path = a.run / "progress.jsonl"
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    stream.seek(offset)
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith("\n"):
                            offset = start
                            break
                        record = json.loads(line)
                        step = record["step"]
                        fields = {
                            **record.get("metrics", {}),
                            **{k: v for k, v in record.items() if k not in {"metrics", "step"}},
                        }
                        for key, value in fields.items():
                            if isinstance(value, (int, float)):
                                writer.add_scalar(f"{record['type']}/{key}", value, step)
                    writer.flush()
            if not a.follow:
                break
            time.sleep(5)


if __name__ == "__main__":
    main()
