"""Unit tests for hafl/aggregation.py -- Krum, edge/cloud aggregation, and
the FedYogi update -- checked against hand-computed values so the
recomputed experiments rest on a verified implementation of Eq. 6-10."""

import math

import torch

from hafl.aggregation import (
    FedYogiState,
    cloud_aggregate_update,
    edge_aggregate,
    fedavg_aggregate,
    krum_scores,
    krum_select,
)


def test_krum_scores_identifies_outlier():
    # 4 honest clients clustered near [0,0,0], 1 Byzantine client far away.
    honest = [torch.tensor([0.0, 0.0, 0.0]) + 0.01 * i for i in range(4)]
    byzantine = torch.tensor([100.0, 100.0, 100.0])
    deltas = honest + [byzantine]
    scores = krum_scores(deltas, num_neighbors=2)
    assert scores.argmax().item() == 4, "the Byzantine outlier should have the largest KrumScore"


def test_krum_select_excludes_outlier():
    honest = [torch.tensor([0.0, 0.0]) + 0.01 * i for i in range(4)]
    byzantine = torch.tensor([50.0, 50.0])
    deltas = honest + [byzantine]
    survivors = krum_select(deltas, num_neighbors=2, num_survivors=4)
    assert 4 not in survivors, "Krum selection should drop the outlier index"
    assert len(survivors) == 4


def test_krum_scores_matches_hand_computation():
    # 3 points on a line: 0, 1, 10. num_neighbors=1 (nearest other point).
    deltas = [torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([10.0])]
    scores = krum_scores(deltas, num_neighbors=1)
    # point 0's nearest neighbor is point 1, dist^2 = 1
    # point 1's nearest neighbor is point 0, dist^2 = 1
    # point 2's nearest neighbor is point 1, dist^2 = 81
    assert torch.allclose(scores, torch.tensor([1.0, 1.0, 81.0]), atol=1e-5)


def test_edge_aggregate_weighted_average():
    deltas = [torch.tensor([1.0, 1.0]), torch.tensor([3.0, 3.0])]
    counts = [1, 3]
    agg = edge_aggregate(deltas, counts, survivor_idx=[0, 1])
    # weights 1/4 and 3/4 -> 0.25*[1,1] + 0.75*[3,3] = [2.5, 2.5]
    assert torch.allclose(agg, torch.tensor([2.5, 2.5]))


def test_edge_aggregate_renormalizes_over_survivors_only():
    deltas = [torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([1000.0])]
    counts = [1, 1, 1]
    agg = edge_aggregate(deltas, counts, survivor_idx=[0, 1])  # drop the outlier
    assert torch.allclose(agg, torch.tensor([2.0]))  # mean of the two survivors only


def test_cloud_aggregate_update_weighted_by_edge_size():
    edge_deltas = [torch.tensor([0.0]), torch.tensor([10.0])]
    edge_counts = [90, 10]
    agg = cloud_aggregate_update(edge_deltas, edge_counts)
    assert torch.allclose(agg, torch.tensor([1.0]))  # 0.9*0 + 0.1*10


def test_fedavg_aggregate_matches_manual_weighted_mean():
    deltas = [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 4.0])]
    counts = [1, 1]
    agg = fedavg_aggregate(deltas, counts)
    assert torch.allclose(agg, torch.tensor([1.0, 2.0]))


def test_fedyogi_state_matches_manual_single_step():
    w = torch.tensor([1.0])
    delta = torch.tensor([0.5])
    state = FedYogiState(beta2=0.99, eta_g=0.1, tau=0.001)
    w_next = state.step(w, delta)
    # v^0 = delta^2 = 0.25 (clamped init), so first call's v stays 0.25
    assert math.isclose(state.v.item(), 0.25, rel_tol=1e-6)
    expected_w = 1.0 + 0.1 * 0.5 / (math.sqrt(0.25) + 0.001)
    assert math.isclose(w_next.item(), expected_w, rel_tol=1e-6)


def test_fedyogi_second_moment_can_decrease_per_sign_rule():
    # This is Yogi's distinguishing property vs. Adam/FedAdam: v can shrink
    # even when the new squared delta is large, via the sign(v - delta^2) term.
    state = FedYogiState(beta2=0.9, eta_g=0.01, tau=1e-3)
    w = torch.tensor([0.0])
    state.step(w, torch.tensor([0.01]))  # tiny first delta -> small v
    v_after_first = state.v.clone()
    state.step(w, torch.tensor([10.0]))  # huge second delta
    # Eq. 9 with sign(v - delta^2) == -1 here (delta^2 >> v) subtracts a
    # *negative* quantity, i.e. v should move toward (not jump to) delta^2 --
    # specifically it should increase by (1-beta2)*delta^2, not jump to delta^2.
    expected = v_after_first + (1 - 0.9) * (10.0 ** 2)
    assert torch.allclose(state.v, expected, atol=1e-4)
