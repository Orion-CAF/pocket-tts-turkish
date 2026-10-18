"""Run before paying for a long training run; fails on missing data or wrong GPU."""

import argparse
import json
import shutil
from pathlib import Path

import sentencepiece as spm
import torch

from training.args import load_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    a = p.parse_args()
    args = load_args(a.config)
    assert torch.cuda.is_available(), "CUDA unavailable; install CUDA-enabled torch"
    assert torch.cuda.is_bf16_supported(), "Native bf16 required for this recipe"
    assert shutil.disk_usage(".").free > 25 * 1024**3, "Reserve >=25 GiB for checkpoints/cache"
    tokenizer = args.model_overrides["flow_lm.lookup_table.tokenizer_path"]
    assert spm.SentencePieceProcessor(model_file=tokenizer).vocab_size() == 4000
    for manifest in [args.data.train_jsonl, args.data.valid_jsonl]:
        count = 0
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            assert Path(e["path"]).is_file(), e["path"]
            assert e.get("words")
            count += 1
        assert count >= args.batch_size * args.data.loader_procs * args.data.num_bucket_batches
    if args.distill_cfg_coef:
        assert Path(args.distill_teacher_weights).is_file(), (
            "Select and copy teacher checkpoint first"
        )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "effective_batch": args.batch_size * args.grad_accum_steps,
                "status": "FILES_OK_FORWARD_BACKWARD_SMOKE_STILL_REQUIRED",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
