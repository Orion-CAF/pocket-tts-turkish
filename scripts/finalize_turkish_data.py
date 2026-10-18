"""Validate complete alignment logs, train tokenizer and write portable manifests."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import sentencepiece as spm

from scripts.align_turkish_local import valid_cuts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/tr/v1"))
    a = p.parse_args()
    root = a.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    report = {}
    corpora = {}
    for split in ["train", "valid", "test"]:
        source = [
            json.loads(l)
            for l in (root / f"{split}_unaligned.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        aligned = [
            json.loads(l)
            for l in (root / f"{split}_alignment_log.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert {e["path"] for e in source} == {e["path"] for e in aligned}, (
            f"{split}: alignment incomplete"
        )
        assert len({e["path"] for e in aligned}) == len(aligned), "Duplicate alignment log entries"
        accepted = []
        rejected: Counter[str] = Counter()
        rejected_hours: Counter[str] = Counter()
        for e in aligned:
            if e["alignment_status"] != "accepted":
                rejected[e["alignment_status"]] += 1
                rejected_hours[e["alignment_status"]] += e["duration"] / 3600
                continue
            if e.get("review_flag"):
                rejected["asr_text_disagreement"] += 1
                rejected_hours["asr_text_disagreement"] += e["duration"] / 3600
                continue
            assert " ".join(w["word"] for w in e["words"]) == e["transcript"]
            previous = 0.0
            for w in e["words"]:
                assert previous <= w["start"] < w["end"] <= e["duration"] + 0.001
                previous = w["end"]
            assert valid_cuts(e["words"], e["duration"])
            path = Path(e["path"]).resolve()
            assert path.is_file()
            e["path"] = path.relative_to(repo).as_posix()
            accepted.append(e)
        assert len(accepted) >= 64, f"Too few accepted {split} examples"
        corpora[split] = accepted
        report[split] = {
            "records": len(accepted),
            "hours": sum(e["duration"] for e in accepted) / 3600,
            "rejected_records": dict(rejected),
            "rejected_hours": dict(rejected_hours),
        }
    for split, entries in corpora.items():
        (root / f"{split}_aligned.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8"
        )
    corpus = root / "tokenizer_train.txt"
    corpus.write_text("\n".join(e["transcript"] for e in corpora["train"]) + "\n", encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(root / "tokenizer"),
        vocab_size=4000,
        model_type="unigram",
        character_coverage=1.0,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        max_sentencepiece_length=6,
        allow_whitespace_only_pieces=True,
        split_digits=True,
        byte_fallback=True,
        pad_id=3,
        num_threads=1,
        shuffle_input_sentence=False,
    )
    sp = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
    assert sp.vocab_size() == 4000
    for text in ["İstanbul ile Iğdır", "ışık, ilgi, İpek, Işıl", "Çığ, öğün ve şüphe."]:
        assert sp.decode(sp.encode(text)) == text
    report["tokenizer_sha256"] = hashlib.sha256((root / "tokenizer.model").read_bytes()).hexdigest()
    report["status"] = "STRUCTURALLY_VALID_REQUIRES_LISTENING_REVIEW"
    (root / "final_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
