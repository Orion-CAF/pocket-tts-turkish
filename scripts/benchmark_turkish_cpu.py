"""Save CPU streaming samples and report first-chunk latency plus real-time factor."""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from pocket_tts import TTSModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--voice", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    model = TTSModel.load_model(config=a.config).to("cpu")
    state = model.get_state_for_audio_prompt(a.voice)
    texts = [
        "Merhaba, bugün Türkçe konuşmayı öğreniyoruz.",
        "İstanbul ile Iğdır arasında uzun bir yol var.",
        "Işıl, İpek ile öğleden sonra görüşecek.",
    ]
    results = []
    for i, text in enumerate(texts):
        torch.manual_seed(42)
        start = time.perf_counter()
        chunks, first = [], None
        for chunk in model.generate_audio_stream(state, text, copy_state=True):
            if first is None:
                first = time.perf_counter() - start
            chunks.append(chunk.detach().cpu().reshape(-1))
        elapsed = time.perf_counter() - start
        assert chunks, "Empty generation"
        audio = torch.cat(chunks)
        duration = len(audio) / model.sample_rate
        sf.write(a.output / f"sample_{i}.wav", audio.numpy(), model.sample_rate)
        results.append(
            {
                "text": text,
                "seconds": duration,
                "generation_seconds": elapsed,
                "first_chunk_seconds": first,
                "rtf": elapsed / duration,
                "cold_first_sample": i == 0,
            }
        )
    (a.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
