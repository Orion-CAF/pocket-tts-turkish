"""Resumable Turkish alignment on CPU or CUDA; conservative, auditable filtering."""

import argparse
import json
import time
import unicodedata
from pathlib import Path
from typing import Any

import jiwer
import soundfile as sf
import torch
from huggingface_hub import HfApi
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from training.scripts.align_data import _tokens_for, batched_word_spans

MODEL = "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish"


def fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(str.maketrans({"I": "ı", "İ": "i"})).lower()


def valid_cuts(words: list[dict[str, Any]], duration: float) -> list[int]:
    return [
        i
        for i in range(1, len(words))
        if words[i - 1]["end"] < 2.0
        and 1.0 <= (words[i - 1]["end"] + words[i]["start"]) / 2 <= duration - 1.0
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tr/v1"))
    p.add_argument(
        "--limit", type=int, default=0, help="New entries per invocation; zero means all"
    )
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install CUDA-enabled torch")
    meta_path = a.root / "alignment_model.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"model": MODEL, "revision": HfApi().model_info(MODEL).sha}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    processor = Wav2Vec2Processor.from_pretrained(MODEL, revision=meta["revision"])
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, revision=meta["revision"]).to(device).eval()
    vocab = processor.tokenizer.get_vocab()
    delim = vocab[processor.tokenizer.word_delimiter_token]
    blank = processor.tokenizer.pad_token_id
    n = 0
    started = time.monotonic()
    for split in ["valid", "train", "test"]:
        target = a.root / f"{split}_alignment_log.jsonl"
        done = set()
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                done.add(json.loads(line)["path"])
        with target.open("a", encoding="utf-8", buffering=1) as output:
            for line in (
                (a.root / f"{split}_unaligned.jsonl").read_text(encoding="utf-8").splitlines()
            ):
                entry = json.loads(line)
                if entry["path"] in done:
                    continue
                if a.limit and n >= a.limit:
                    return
                words = entry["transcript"].split()
                normalized = []
                reason = None
                for word in words:
                    normalized_word = "".join(c for c in fold(word) if c.isalpha())
                    if not normalized_word or any(c not in vocab for c in normalized_word):
                        reason = "unsupported_word"
                    normalized.append(normalized_word)
                if not reason:
                    wav, sr = sf.read(entry["path"], dtype="float32")
                    if sr != processor.feature_extractor.sampling_rate:
                        raise ValueError(f"Unexpected sample rate {sr}")
                    inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
                    with torch.inference_mode():
                        # The neural forward runs on GPU; the small sequential trellis
                        # stays on CPU to avoid hundreds of tiny GPU launches.
                        emissions = model(**inputs).logits.float().log_softmax(-1).cpu()
                    hypothesis = processor.batch_decode(emissions.argmax(-1))[0]
                    entry["alignment_asr_text"] = hypothesis
                    ref = " ".join(normalized)
                    hyp = " ".join(
                        "".join(c if c.isalpha() else " " for c in fold(hypothesis)).split()
                    )
                    entry["alignment_asr_wer"] = jiwer.wer(ref, hyp)
                    entry["review_flag"] = entry["alignment_asr_wer"] > 0.5
                    tokens, word_of = _tokens_for(normalized, vocab, delim)
                    frames = emissions.shape[1]
                    spans = batched_word_spans(
                        emissions, torch.tensor([frames]), [tokens], [word_of], blank
                    )[0]
                    if spans is None or len(spans) != len(words) or any(s is None for s in spans):
                        reason = "unalignable"
                    else:
                        scale = len(wav) / sr / frames
                        timed = [
                            {
                                "word": w,
                                "start": round(s * scale, 4),
                                "end": round((e + 1) * scale, 4),
                            }
                            for w, (s, e) in zip(words, spans, strict=True)
                        ]
                        if not valid_cuts(timed, entry["duration"]):
                            reason = "no_valid_reference_target_cut"
                        else:
                            entry["words"] = timed
                entry["alignment_status"] = reason or "accepted"
                output.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n += 1
                if n % 20 == 0:
                    print(
                        f"{split}: {n} new entries; {(time.monotonic() - started) / n:.2f} s/entry",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
