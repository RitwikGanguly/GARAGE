# Generative Adversarial Networks in GARAGE

## The GAN Framework

A Generative Adversarial Network (GAN) is a framework for estimating generative models via an adversarial process. Two neural networks — a **Generator** ($G$) and a **Discriminator** ($D$) — are trained simultaneously in a minimax game:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

**Intuition:** The generator produces counterfeit data. The discriminator is the detective trying to detect counterfeits. Both improve together until the counterfeits become indistinguishable from real data.

### Generator ($G$)

- **Input:** Random noise vector $z \sim \mathcal{N}(0, I)$ (or a hybrid batch in GARAGE).
- **Output:** Synthetic data sample $\tilde{x} = G(z)$ with the same dimensions as real data.
- **Goal:** Maximise $D(G(z))$, i.e., fool the discriminator.

### Discriminator ($D$)

- **Input:** A data sample — either real ($x$) or generated ($G(z)$).
- **Output:** A scalar probability $D(\cdot) \in [0,1]$ that the input is real.
- **Goal:** $D(x) \to 1$ for real data, $D(G(z)) \to 0$ for generated data.

### Loss Functions

**Generator Loss:**

$$L_G = -\log D(G(z))$$

**Discriminator Loss:**

$$L_D = -\log D(x) - \log(1 - D(G(z)))$$

These create an adversarial dynamic: the generator minimises $L_G$ while the discriminator maximises its ability to distinguish real from fake.

---

## How GARAGE Uses GANs

### The Hybrid Input Batch

Unlike standard GANs, GARAGE's generator does not receive pure random noise. Instead, it receives a **hybrid input batch**:

$$Z_{\text{batch}} = (1 - \lambda) \cdot z_{\text{noise}} \; \oplus \; \lambda \cdot x_{\text{seed}}$$

where:
- $z_{\text{noise}} \sim \mathcal{N}(0, I)$ are random noise vectors,
- $x_{\text{seed}}$ are **real cells selected by the GAT** (the most representative cells),
- $\lambda \in [0, 1]$ is the **leakage fraction** (default: 0.2).

This "attention-guided leakage" serves two purposes:

1. **Stabilises training:** The generator is anchored to real, biologically meaningful states, reducing mode collapse and training instability.
2. **Preserves rare cells:** Seed cells are drawn proportionally from all cell types (with extra weight on rare types), so the generator learns to produce them.

### Architecture Details

**Generator (MLP):**

```
z_batch (n_features)  →  Linear(1024) → ReLU → Linear(1024) → ReLU → Linear(n_features) → Sigmoid
```

**Discriminator (MLP):**

```
data (n_features) → Linear(512) → ReLU → Linear(256) → ReLU → Linear(1)
```

Both use fully connected layers with ReLU activations. The generator uses Sigmoid on the output layer to produce normalised expression-like values.

### Training Loop

For each training iteration ($T = 20{,}001$ by default):

1. **Discriminator step** (× `nd_steps` = 5): Train on a batch of real data (label 0.9) and a batch of generated data (label 0.1), using label smoothing for stability.
2. **Generator step** (× `ng_steps` = 2): Sample new noise + GAT seeds to form the hybrid batch, generate fake data, and compute the generator loss against the discriminator's judgement.

All hyper-parameters are configured in `config.py` via `GARAGE_DEFAULTS`.

---

## Advantages of GANs for scRNA-seq Data

| Property | Benefit |
|----------|---------|
| **Implicit modelling** | No need to explicitly model the probability distribution of high-dimensional gene expression data. |
| **Diversity** | Can generate a wide range of cellular states, not just "average" cells. |
| **Scalability** | Trains on a subset, generates arbitrarily many synthetic cells. |
| **Privacy preservation** | Synthetic cells do not correspond to real individuals. |

---

## Common Challenges and GARAGE's Countermeasures

| Challenge | Standard GAN | GARAGE Solution |
|-----------|-------------|----------------|
| **Mode collapse** | Generator produces one or few cell types repeatedly. | GAT seeding anchors the generator to diverse seed cells. |
| **Training instability** | Oscillating or diverging losses. | Leakage fraction $\lambda$ provides a stable signal. |
| **Rare cell dropout** | Rare types vanish — their signal is overwhelmed by abundant types. | GAT priority weight boosts rare-type cells in the seed batch. |
| **Evaluation difficulty** | No single metric captures quality. | Multi-metric validation: WD, MMD, ARI, NMI, F1, UMAP. |

---

## Variants Used in Benchmarking

GARAGE ships PyTorch implementations of five general-purpose GAN variants for comparison:

| Model | Key Innovation |
|-------|---------------|
| `gan.py` | Standard BCE GAN |
| `wgan.py` | Wasserstein GAN (RMSprop, weight clipping) |
| `fgan.py` | f-divergence GAN (Fisher ratio, constraint penalty) |
| `vae.py` | Variational Autoencoder (MSE reconstruction + KL divergence) |
| `lsh_gan.py` | LSH-GAN (random KNN subsample + GAN) |

These are in `benchmarking/sota/`. See also `benchmarking/scrna_seq_specific/` for scRNA-seq domain-specific baselines (scGAN, scVAE, scDiffusion, GAN-ROS, VAE-ROS).
