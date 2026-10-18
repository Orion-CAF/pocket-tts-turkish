"""Download, audit and export Common Voice; alignment remains a separate stage."""

import argparse
import hashlib
import io
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

DATASET = "ysdede/commonvoice_17_tr_fixed"
REVISION = "9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b"


def text_key(text: str) -> str:
    text = text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/tr/v1"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free disk is required before starting")
    root = Path(
        snapshot_download(
            DATASET,
            repo_type="dataset",
            revision=REVISION,
            local_dir=out / "source",
            allow_patterns=["README.md", "data/*.parquet"],
        )
    )
    counts = Counter()
    hours = Counter()
    held_audio, held_text, seen_audio = set(), set(), set()
    report = out / "audit.jsonl"
    with report.open("w", encoding="utf-8") as audit:
        for split in ["test", "validation", "train", "validated"]:
            files = sorted((root / "data").glob(split + "-*.parquet"))
            if not files:
                raise RuntimeError(f"Missing split: {split}")
            destination = {"validation": "valid", "validated": "train"}.get(split, split)
            mode = "a" if split == "validated" else "w"
            with (out / f"{destination}_unaligned.jsonl").open(mode, encoding="utf-8") as manifest:
                index = 0
                for file in files:
                    for batch in pq.ParquetFile(file).iter_batches(batch_size=64):
                        for record in batch.to_pylist():
                            index += 1
                            item = {"split": split, "index": index}
                            counts[f"{split}.total"] += 1
                            try:
                                wav, sr = sf.read(
                                    io.BytesIO(record["audio"]["bytes"]),
                                    dtype="float32",
                                    always_2d=True,
                                )
                                if not len(wav) or not np.isfinite(wav).all():
                                    raise ValueError("invalid_audio")
                                duration = len(wav) / sr
                                digest = hashlib.sha256(
                                    str((sr, wav.shape)).encode() + wav.tobytes()
                                ).hexdigest()
                                text = " ".join(
                                    unicodedata.normalize("NFC", record["transcription"]).split()
                                )
                                key = text_key(text)
                                item.update(duration=duration, audio_hash=digest, transcript=text)
                                hours[f"{split}.decoded"] += duration / 3600
                                overlap = digest in held_audio or key in held_text
                                if split in {"test", "validation"}:
                                    held_audio.add(digest)
                                    held_text.add(key)
                                reason = None
                                if not 3 <= duration <= 10:
                                    reason = "duration_outside_3_10"
                                elif len(text.split()) < 2 or not key:
                                    reason = "insufficient_text"
                                elif any(c.isdigit() for c in text):
                                    reason = "number_requires_manual_normalization"
                                elif split in {"train", "validated"} and overlap:
                                    reason = "heldout_overlap"
                                elif digest in seen_audio:
                                    reason = "duplicate_audio"
                                elif float(np.max(np.abs(wav))) < 0.001:
                                    reason = "near_silent"
                                if reason:
                                    item["status"] = reason
                                else:
                                    if shutil.disk_usage(out).free < 2 * 1024**3:
                                        raise OSError("Disk reserve below 2 GiB; stopping")
                                    relative = Path("audio") / split / f"{index:06d}.flac"
                                    path = out / relative
                                    path.parent.mkdir(parents=True, exist_ok=True)
                                    sf.write(path, wav.mean(axis=1), sr, subtype="PCM_16")
                                    entry = {
                                        "path": path.as_posix(),
                                        "duration": duration,
                                        "transcript": text,
                                        "source_split": split,
                                        "source_index": index,
                                        "audio_hash": digest,
                                    }
                                    manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                    seen_audio.add(digest)
                                    item["status"] = "candidate"
                                    hours[f"{split}.candidate"] += duration / 3600
                            except OSError:
                                raise
                            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                                item["status"] = "decode_or_schema_error"
                                item["error"] = str(exc)
                            counts[f"{split}.{item['status']}"] += 1
                            audit.write(json.dumps(item, ensure_ascii=False) + "\n")
                            if index % 2000 == 0:
                                print(split, index, "processed", flush=True)
            print(split, dict(counts), flush=True)
    summary = {
        "dataset": DATASET,
        "revision": REVISION,
        "status": "UNALIGNED_NOT_TRAIN_READY",
        "counts": dict(counts),
        "hours": dict(hours),
        "normalization": "NFC + whitespace; numbers excluded",
        "speaker_disjoint": False,
        "audio": "mono PCM16 FLAC at original sample rate",
    }
    (out / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
