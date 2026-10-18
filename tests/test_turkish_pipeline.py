"""Small contract tests for Turkish text, alignment cuts and config compatibility."""

import unittest

import torch

from scripts.align_turkish_local import fold, valid_cuts
from scripts.evaluate_turkish import normalize
from training.args import load_args
from training.scripts.align_data import batched_word_spans


class TurkishContracts(unittest.TestCase):
    def test_dotted_and_dotless_i(self):
        self.assertEqual(fold("IĞDIR İSTANBUL"), "ığdır istanbul")
        self.assertEqual(normalize("İpek, Işıl! Çığ?"), "ipek ışıl çığ")

    def test_no_overlapping_voice_target_fallback(self):
        words = [{"start": 0.1, "end": 0.9}, {"start": 1.1, "end": 1.8}]
        self.assertEqual(valid_cuts(words, 2.1), [1])
        self.assertEqual(valid_cuts(words, 1.8), [])

    def test_known_alignment(self):
        emissions = torch.full((1, 5, 3), -20.0)
        for frame, token in enumerate([0, 1, 0, 2, 0]):
            emissions[0, frame, token] = 0
        spans = batched_word_spans(emissions, torch.tensor([5]), [[1, 2]], [[0, 1]], 0)
        self.assertEqual(spans, [[(1, 1), (3, 3)]])

    def test_teacher_student_share_tokenizer(self):
        teacher = load_args("training/configs/turkish_vast_teacher.yaml")
        student = load_args("training/configs/turkish_vast_student.yaml")
        self.assertTrue(teacher.reset_text_embedding)
        self.assertFalse(student.reset_text_embedding)
        key = "flow_lm.lookup_table.tokenizer_path"
        self.assertEqual(teacher.model_overrides[key], student.model_overrides[key])
        self.assertEqual(student.model_overrides[key], student.distill_teacher_overrides[key])
        self.assertEqual(student.model_overrides["flow_lm.transformer.num_layers"], 6)
        self.assertEqual(teacher.batch_size * teacher.grad_accum_steps, 64)
        self.assertEqual(student.batch_size * student.grad_accum_steps, 64)

    def test_student8_differs_from_student_only_in_depth(self):
        student = load_args("training/configs/turkish_vast_student.yaml")
        student8 = load_args("training/configs/turkish_vast_student8.yaml")
        self.assertEqual(student8.model_overrides["flow_lm.transformer.num_layers"], 8)
        self.assertNotEqual(student8.run_dir, student.run_dir)
        student8.model_overrides["flow_lm.transformer.num_layers"] = 6
        student8.run_dir = student.run_dir
        self.assertEqual(student8, student)

    def test_student_ema_forgets_its_seed(self):
        # Evaluation and export read the EMA weights; by the end of the run the
        # seed's share of them (decay ** max_steps) must be negligible.
        student = load_args("training/configs/turkish_vast_student.yaml")
        self.assertLess(student.ema_decay**student.max_steps, 0.05)


if __name__ == "__main__":
    unittest.main()
