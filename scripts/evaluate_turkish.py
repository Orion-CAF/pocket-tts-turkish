"""Deterministic held-out continuation TTS evaluation with Turkish WER/CER.

Use validation during development, test only for the final comparison. Also scores
the original target audio to expose the ASR error floor. No speaker-disjoint claim.
"""

import argparse
import gc
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from huggingface_hub import HfApi
from transformers import pipeline

from scripts.align_turkish_local import valid_cuts
from training.eval.librispeech import latents_to_wav, load_mono, load_run


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    return " ".join(re.findall(r"\w+", text))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--manifest", type=Path, default=Path("data/tr/v1/valid_aligned.jsonl"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cfg", type=float, required=True, help="Teacher 2; distilled student 1")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--asr", default="openai/whisper-small")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    items = [json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines()]
    random.Random(42).shuffle(items)
    items = items[: a.limit]
    torch.manual_seed(42)
    device = torch.device(a.device)
    model, mimi, step = load_run(a.run, device, use_ema=True, checkpoint=a.checkpoint)
    records = []
    for idx, entry in enumerate(items):
        words = entry["words"]
        i = valid_cuts(words, entry["duration"])[0]
        cut = (words[i - 1]["end"] + words[i]["start"]) / 2
        text = " ".join(w["word"] for w in words[i:])
        wav = load_mono(entry["path"], mimi.sample_rate)
        split = round(cut * mimi.sample_rate)
        with torch.inference_mode():
            voice = mimi.encode_to_latent(wav[:split][None, None].to(device))[0]
            tokens = torch.tensor(
                model.flow_lm.conditioner.tokenizer.encode(text), dtype=torch.long
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.monotonic()
            latent = model.generate(
                [tokens],
                [voice],
                max_frames=250,
                temp=0.3,
                n_steps=1,
                cfg_coef=a.cfg,
                eos_threshold=0.0,
            )[0]
            audio = latents_to_wav(mimi, latent, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.monotonic() - started
        generated = a.output / f"{idx:04d}_generated.wav"
        real = a.output / f"{idx:04d}_real.wav"
        sf.write(real, wav[split:].numpy(), mimi.sample_rate)
        if audio is not None:
            sf.write(generated, audio.detach().cpu().numpy(), mimi.sample_rate)
        records.append(
            {
                "id": idx,
                "source": entry["path"],
                "text": text,
                "reference": normalize(text),
                "generated": str(generated),
                "real": str(real),
                "silent": audio is None,
                "capped": len(latent) >= 250,
                "generation_seconds": seconds,
                "audio_seconds": 0 if audio is None else len(audio) / mimi.sample_rate,
            }
        )
        if (idx + 1) % 10 == 0 or idx + 1 == len(items):
            print(f"generated {idx + 1}/{len(items)}", flush=True)
    del model, mimi
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    revision = HfApi().model_info(a.asr).sha
    asr = pipeline(
        "automatic-speech-recognition",
        model=a.asr,
        revision=revision,
        device=a.device,
        torch_dtype=torch.float32,
    )
    from scipy.signal import resample_poly

    def transcribe(path: str) -> str:
        wav, sr = sf.read(path, dtype="float32")
        if sr != 16000:
            import math

            divisor = math.gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // divisor, sr // divisor)
        return normalize(
            asr(
                {"array": wav, "sampling_rate": 16000},
                generate_kwargs={"language": "turkish", "task": "transcribe"},
            )["text"]
        )

    with (a.output / "records.jsonl").open("w", encoding="utf-8", buffering=1) as f:
        for idx, r in enumerate(records):
            r["hypothesis"] = "" if r["silent"] else transcribe(r["generated"])
            r["real_hypothesis"] = transcribe(r["real"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (idx + 1) % 10 == 0 or idx + 1 == len(records):
                print(f"transcribed {idx + 1}/{len(records)}", flush=True)
    refs = [r["reference"] for r in records]
    metrics = {
        "wer": jiwer.wer(refs, [r["hypothesis"] for r in records]),
        "cer": jiwer.cer(refs, [r["hypothesis"] for r in records]),
        "real_audio_wer": jiwer.wer(refs, [r["real_hypothesis"] for r in records]),
        "real_audio_cer": jiwer.cer(refs, [r["real_hypothesis"] for r in records]),
        "silent": sum(r["silent"] for r in records),
        "capped": sum(r["capped"] for r in records),
    }
    result = {
        "step": step,
        "checkpoint": a.checkpoint,
        "manifest": str(a.manifest),
        "asr": a.asr,
        "asr_revision": revision,
        "cfg": a.cfg,
        "seed": 42,
        "items": len(records),
        "metrics": metrics,
    }
    (a.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(str(a.run / "tensorboard")) as writer:
        for key, value in metrics.items():
            writer.add_scalar(f"eval/{a.manifest.stem}_n{len(records)}/{key}", value, step)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
