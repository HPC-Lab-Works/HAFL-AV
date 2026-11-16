"""Unit tests for hafl/topology.py: edge/vehicle indexing and the Dirichlet
non-IID partition used for the data-heterogeneity sensitivity sweep."""

import numpy as np

from hafl.topology import build_topology, dirichlet_label_partition


def test_topology_vehicle_indexing():
    topo = build_topology(num_edges=3, vehicles_per_edge=4)
    assert topo.num_vehicles == 12
    assert topo.edge_of(0) == 0
    assert topo.edge_of(3) == 0
    assert topo.edge_of(4) == 1
    assert topo.edge_of(11) == 2
    assert topo.vehicle_ids_for_edge(1) == [4, 5, 6, 7]


def test_topology_uneven_vehicles_per_edge():
    topo = build_topology(num_edges=2, vehicles_per_edge=[3, 5])
    assert topo.num_vehicles == 8
    assert topo.vehicle_ids_for_edge(0) == [0, 1, 2]
    assert topo.vehicle_ids_for_edge(1) == [3, 4, 5, 6, 7]


def test_dirichlet_partition_covers_every_sample_exactly_once():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=2000)
    parts = dirichlet_label_partition(labels, num_clients=10, alpha=0.5, seed=0)
    all_idx = np.concatenate(parts)
    assert len(all_idx) == len(labels)
    assert set(all_idx.tolist()) == set(range(len(labels)))  # every sample assigned, none duplicated


def test_dirichlet_partition_more_skewed_at_low_alpha():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, size=4000)

    def class_entropy_of_client(client_idx, client_labels):
        counts = np.bincount(labels[client_labels], minlength=4).astype(float)
        p = counts / counts.sum()
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    low_alpha_parts = dirichlet_label_partition(labels, num_clients=20, alpha=0.05, seed=2)
    high_alpha_parts = dirichlet_label_partition(labels, num_clients=20, alpha=100.0, seed=2)

    low_entropy = np.mean([class_entropy_of_client(i, p) for i, p in enumerate(low_alpha_parts)])
    high_entropy = np.mean([class_entropy_of_client(i, p) for i, p in enumerate(high_alpha_parts)])
    assert low_entropy < high_entropy, "low alpha (severe non-IID) should yield lower per-client label entropy"


def test_dirichlet_partition_min_samples_floor():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 3, size=300)
    parts = dirichlet_label_partition(labels, num_clients=30, alpha=0.01, seed=3, min_samples_per_client=5)
    assert all(len(p) >= 5 for p in parts)
