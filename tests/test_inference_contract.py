import pytest


def test_inference_module_is_importable_without_loading_a_checkpoint():
    from xnlp_trainer.inference import XNLPPredictor
    assert hasattr(XNLPPredictor, "load")
    assert hasattr(XNLPPredictor, "generate_candidates")


def test_empty_generation_prompt_is_rejected():
    from xnlp_trainer.inference import XNLPPredictor
    with pytest.raises(ValueError):
        XNLPPredictor.generate(None, "")  # type: ignore[arg-type]


def test_predictor_exposes_grounded_structure_and_answer_methods():
    from xnlp_trainer.inference import XNLPPredictor
    predictor = object.__new__(XNLPPredictor)
    structure = predictor.understand("Umntu uyafunda.")
    answer = predictor.answer("UMqhayi ngumbongi.", "Ngubani uMqhayi?")
    assert structure["schema_version"] == 1
    assert answer["status"] == "SUPPORTED_EXTRACTION"
    assert predictor.summarize("Umntu uyafunda. Ubuntu buhle.")["summary"] == "Umntu uyafunda. Ubuntu buhle."
