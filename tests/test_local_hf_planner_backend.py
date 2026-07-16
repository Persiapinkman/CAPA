from training.planner_grpo_seed_v1.scripts.local_hf_planner_backend import (
    LocalHFPlannerBackend,
    env_bool,
)
from util.vlm_service import omit_model_image_payload


def test_env_bool_is_explicit(monkeypatch):
    monkeypatch.delenv("LOCAL_TEST_FLAG", raising=False)
    assert env_bool("LOCAL_TEST_FLAG") is False
    monkeypatch.setenv("LOCAL_TEST_FLAG", "yes")
    assert env_bool("LOCAL_TEST_FLAG") is True
    monkeypatch.setenv("LOCAL_TEST_FLAG", "off")
    assert env_bool("LOCAL_TEST_FLAG") is False


def test_local_backend_service_factory_reuses_loaded_model():
    backend = object.__new__(LocalHFPlannerBackend)
    assert backend.service_factory(api_key="ignored", base_url="ignored") is backend


def test_image_payload_omission_is_opt_in(monkeypatch):
    monkeypatch.delenv("CAPA_OMIT_MODEL_IMAGE_PAYLOAD", raising=False)
    assert omit_model_image_payload() is False
    monkeypatch.setenv("CAPA_OMIT_MODEL_IMAGE_PAYLOAD", "true")
    assert omit_model_image_payload() is True
