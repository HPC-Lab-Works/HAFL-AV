"""Unit tests for hafl/privacy.py: the Eq. (5) clip+noise operator and the
epsilon accountant (exercised against whichever backend -- opacus or the
analytic fallback -- is installed in the current environment)."""

import torch

from hafl.privacy import clip_and_noise, compute_epsilon, solve_noise_multiplier


def test_clip_and_noise_respects_clip_norm_in_expectation():
    torch.manual_seed(0)
    big_delta = torch.full((1000,), 100.0)  # norm way above clip_norm
    clipped = clip_and_noise(big_delta, clip_norm=1.0, noise_multiplier=0.0)
    assert abs(clipped.norm().item() - 1.0) < 1e-3


def test_clip_and_noise_is_a_no_op_below_clip_norm_with_zero_noise():
    small_delta = torch.tensor([0.1, 0.0, 0.0])
    out = clip_and_noise(small_delta, clip_norm=10.0, noise_multiplier=0.0)
    assert torch.allclose(out, small_delta)


def test_clip_and_noise_adds_noise_of_expected_scale():
    torch.manual_seed(0)
    delta = torch.zeros(20000)
    out = clip_and_noise(delta, clip_norm=2.0, noise_multiplier=0.5)
    # sigma = noise_multiplier * clip_norm = 1.0; sample std should be close
    empirical_std = out.std().item()
    assert 0.9 < empirical_std < 1.1


def test_epsilon_decreases_as_noise_multiplier_increases():
    eps_low_noise = compute_epsilon(noise_multiplier=0.5, delta=1e-5, sample_rate=1.0, rounds=50)
    eps_high_noise = compute_epsilon(noise_multiplier=5.0, delta=1e-5, sample_rate=1.0, rounds=50)
    assert eps_high_noise < eps_low_noise


def test_solve_noise_multiplier_roundtrips_to_target_epsilon():
    target = 4.0
    sigma = solve_noise_multiplier(target_epsilon=target, delta=1e-5, sample_rate=1.0, rounds=50)
    achieved = compute_epsilon(noise_multiplier=sigma, delta=1e-5, sample_rate=1.0, rounds=50)
    assert abs(achieved - target) / target < 0.1  # within 10% of target
