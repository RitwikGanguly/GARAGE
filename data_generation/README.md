# Data Generation

Core GARAGE pipeline for synthetic scRNA-seq data generation.

## Files

| File | Description |
|---|---|
| `garage.py` | Main GARAGE pipeline: GAT subsampling (Stage 1) + GAN generation with attention-guided seeding (Stage 2). |
| `wasserstein_distance.py` | Computes Earth Mover's Distance (1-Wasserstein) between real and generated distributions using the POT library. |

## Usage

```bash
python -m data_generation.garage --dataset muraro
python -m data_generation.wasserstein_distance --dataset muraro --gen_csv <path_to_gen_csv>
```
