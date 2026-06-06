# How-to: Benchmarking

A recipe for running all models and building comparison tables.

## Goal

Compare GARAGE against 10 baselines across 4 datasets and produce mean ± std summary tables.

## Prerequisites

- `requirements_benchmarking.txt` installed.
- GARAGE-generated data for all 4 datasets.

## Steps

### 1. Generate data from all models

```bash
# GARAGE
for d in yan pollen cbmc muraro; do
    python -m data_generation.garage --dataset $d
done

# SOTA baselines (5 models)
for d in yan pollen cbmc muraro; do
    python -m benchmarking.sota.gan --dataset $d
    python -m benchmarking.sota.wgan --dataset $d
    python -m benchmarking.sota.fgan --dataset $d
    python -m benchmarking.sota.vae --dataset $d
    python -m benchmarking.sota.lsh_gan --dataset $d
done
```

### 2. Compute metrics for all

```bash
python analysis/distribution_metrics.py
python analysis/clustering_evaluation.py
```

### 3. Build tables

```bash
python analysis/aggregate_losses.py
python analysis/build_summary_tables.py
```

### 4. Check the output

```bash
cat results/summary_wasserstein.csv
cat results/summary_ari_nmi_f1.csv
```

These CSVs are ready for import into R, Python, or Excel for figure generation.

## Related

- [Tutorial: Benchmark Against SOTA](tutorials/04_benchmark_against_sota) — full walkthrough with interpretation.
