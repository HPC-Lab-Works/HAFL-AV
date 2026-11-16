# HAFL-AV

Hierarchical Adaptive Federated Learning for Autonomous Vehicles.

HAFL-AV is a two-tier federated learning framework for training perception models (object detection) across fleets of vehicles without centralizing raw sensor data. Vehicles train locally and send updates to edge servers; edge servers apply Krum-based Byzantine-robust filtering and aggregate within their group; a cloud server aggregates across edges and applies an adaptive (FedYogi-style) global update. Differential privacy noise can be applied to client updates before aggregation.

## Repository structure

- `hafl/` — core library
  - `methods/` — `hafl_av.py` (the full hierarchical method) and flat baselines (`fedavg.py`, `fedprox.py`, `fedyogi.py`, `scaffold.py`)
  - `aggregation.py` — Krum scoring and selection, edge and cloud aggregation, FedYogi state
  - `topology.py` — client-to-edge assignment and Dirichlet label-skew partitioning
  - `client.py` — local client training round
  - `privacy.py` — differential privacy clipping, noise calibration, and epsilon accounting
  - `models.py` — model state-dict utilities and the detector architecture
  - `distributed.py`, `comm.py`, `custom_trainers.py`, `metrics.py`, `utils.py` — supporting infrastructure
- `scripts/run_experiment.py` — experiment entry point and configuration builder
- `data/` — KITTI download and partitioning scripts
- `tests/` — unit tests for aggregation, topology, models, privacy, and federated continuity
- `Dockerfile` — containerized environment
- `configs/` — base and per-method experiment configuration

## Setup

Build and run the provided Docker image:

```bash
docker build -t hafl-av .
docker run --gpus all -it --name hafl-av-run -v $(pwd):/workspace/hafl_av hafl-av
```

## Preparing data

```bash
python data/download_kitti.sh
python data/prepare_kitti.py
python data/partition.py
```

This produces the Dirichlet label-skew partition (default: 5 edges, 6 vehicles per edge, alpha = 0.5) used in the experiments.

## Running an experiment

```bash
python scripts/run_experiment.py --method hafl_av --num_edges 5 --krum_num_survivors 4 \
  --adaptive_momentum true --dp_enabled true
```

Individual mechanisms can be disabled for ablation:

| Mechanism | Disable with |
|---|---|
| Hierarchy (collapse to flat averaging) | `--num_edges 1` |
| Krum filtering | `--krum_num_survivors <vehicles_per_edge>` |
| Adaptive (FedYogi) cloud update | `--adaptive_momentum false` |
| Differential privacy | `--dp_enabled false` |

Flat baselines (`fedavg`, `fedprox`, `fedyogi`, `scaffold`) are run through the same entry point with `--method <name>`.

## Current status

The hierarchical adaptive aggregation path (HAFL-AV with the FedYogi-style cloud update) currently collapses to near-zero mAP50 on KITTI. This has been investigated directly: reducing the global learning rate by two orders of magnitude, and disabling adaptive momentum, Krum filtering, or differential privacy individually, each still collapses, ruling out learning-rate magnitude and any single mechanism as the sole cause. The precise mechanism has not yet been identified; this is documented as an open finding in the accompanying manuscript, and reproducing or investigating it further is welcome.

## License

MIT. See `LICENSE`.
