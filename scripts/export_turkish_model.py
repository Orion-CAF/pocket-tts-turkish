"""Export a selected EMA checkpoint plus tokenizer and portable inference config."""

import argparse
import shutil
from pathlib import Path

import torch
import yaml

from training.args import load_args
from training.checkpointing import export_pocket_safetensors
from training.eval.librispeech import load_run
from training.modules.builders import load_model_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    args = load_args(a.run / "args.yaml")
    if args.distill_cfg_coef <= 0:
        raise ValueError("This public-API package targets the distilled student (CFG=1)")
    model, mimi, step = load_run(a.run, torch.device("cpu"), use_ema=True, checkpoint=a.checkpoint)
    export_pocket_safetensors(a.output / "model.safetensors", model.flow_lm, mimi)
    config = load_model_config(args.model_config, args.model_overrides).model_dump()
    config["weights_path"] = str(a.output / "model.safetensors")
    config["weights_path_without_voice_cloning"] = None
    config["capitalize_first_letter"] = False
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(a.output / "tokenizer.model")
    shutil.copyfile(
        args.model_overrides["flow_lm.lookup_table.tokenizer_path"], a.output / "tokenizer.model"
    )
    (a.output / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
    )
    (a.output / "README.md").write_text(
        f"# Turkish Pocket TTS student\n\nEMA checkpoint step {step}.\n"
        "Paths in config.yaml are relative to the repository working directory; "
        "update the two paths if moving this package. Source audio is Common Voice 16 kHz. "
        "24 kHz output does not restore source high-frequency detail.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
