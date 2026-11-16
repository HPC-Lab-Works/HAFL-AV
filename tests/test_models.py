"""Unit tests for hafl/models.py's state_dict <-> flat-vector plumbing,
which every piece of the aggregation math (Krum, edge/cloud aggregate,
FedYogi) depends on being lossless and shape-exact."""

import torch

from hafl.models import apply_delta, state_dict_delta, state_dict_to_vector, vector_to_state_dict


def _sample_state_dict():
    return {
        "layer1.weight": torch.randn(4, 3),
        "layer1.bias": torch.randn(4),
        "layer2.weight": torch.randn(2, 4),
    }


def test_state_dict_roundtrip_is_lossless():
    sd = _sample_state_dict()
    vec, layout = state_dict_to_vector(sd)
    sd2 = vector_to_state_dict(vec, layout)
    assert sd.keys() == sd2.keys()
    for k in sd:
        assert torch.allclose(sd[k], sd2[k])
        assert sd[k].shape == sd2[k].shape


def test_state_dict_delta_matches_manual_subtraction():
    sd_old = _sample_state_dict()
    sd_new = {k: v + 1.0 for k, v in sd_old.items()}
    delta, layout = state_dict_delta(sd_new, sd_old)
    reconstructed = vector_to_state_dict(delta, layout)
    for k in sd_old:
        assert torch.allclose(reconstructed[k], torch.ones_like(sd_old[k]))


def test_apply_delta_inverts_state_dict_delta():
    sd_old = _sample_state_dict()
    sd_new = {k: v * 2 for k, v in sd_old.items()}
    delta, layout = state_dict_delta(sd_new, sd_old)
    recovered = apply_delta(sd_old, delta, layout)
    for k in sd_old:
        assert torch.allclose(recovered[k], sd_new[k], atol=1e-5)
