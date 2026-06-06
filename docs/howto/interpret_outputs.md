# How-to: Interpret Outputs

A guide to reading GARAGE's multi-metric evaluation and diagnosing issues.

## Goal

Understand what each metric tells you, how to combine them into a coherent story, and how to diagnose common failure modes.

## The Metric Pyramid

Interpret metrics in this order:

```
                    ┌──────────────┐
                    │  UMAP visual │  ← Quick sanity check
                    ├──────────────┤
                    │  WD, MMD     │  ← Distributional fidelity
                    ├──────────────┤
                    │  ARI, NMI    │  ← Clustering quality
                    ├──────────────┤
                    │  macro-F1    │  ← Rare-cell preservation check
                    ├──────────────┤
                    │  Bio. valid. │  ← Biological interpretability
                    └──────────────┘
```

---

## Scenario 1: Everything Looks Good

```
WD = 0.007  (excellent)
ARI = 0.67  (good)
NMI = 0.75  (good)
F1 = 0.61  (reasonable, on par with real data)
UMAP shows separated clusters matching real data
```

**Conclusion:** GARAGE is working well. Proceed to biological validation and rare-cell experiments.

---

## Scenario 2: Good WD, Poor Clustering

```
WD = 0.008  (excellent)
ARI = 0.21  (poor)
NMI = 0.30  (poor)
F1 = 0.15  (poor)
UMAP is a single blob
```

**Diagnosis:** The generator produces data that is *distributionally* similar to the real data (low WD) but lacks the *structure* to form distinct clusters. This is common when:
- The leakage fraction is too high (generator copies real data → WD looks good, but no novelty).
- The generated data has the right gene expression means but wrong covariance structure.

**Fix:**
- Reduce `leakage_fraction` (0.2 → 0.15 or 0.1).
- Try different feature selection methods (`--method pca`).
- Check that the GAN converged (D_loss should be ~ 0.69 plateau).

---

## Scenario 3: Poor WD, Good Clustering

```
WD = 0.34  (poor)
ARI = 0.58  (good)
NMI = 0.65  (good)
F1 = 0.49  (moderate)
UMAP shows decent clusters but shifted relative to real data
```

**Diagnosis:** The generator produces data that clusters nicely but is *shifted* in gene-expression space. This can happen with:
- Under-normalisation (generated and real data on different scales).
- Batch-effect-like translation.

**Fix:**
- Check that the real data and generated data are on the same scale.
- Try re-normalising both to z-scores before computing WD.
- WD is computed on raw normalised data — try computing on PC-reduced data instead.

---

## Scenario 4: Good ARI, Low macro-F1

```
ARI = 0.65  (good)
NMI = 0.72  (good)
F1 = 0.28  (poor)
```

**Diagnosis:** The generated data has good overall cluster structure but **fails to capture rare cell types**. The macro-F1 averages per-type F1 with equal weight, so a low macro-F1 means at least one rare type is being lost.

**Fix:**
- Increase `priority_weight` (2.0 → 4.0).
- Increase `leakage_fraction` slightly.
- Check that `rare_threshold` is set correctly — the rare type may not be flagged as rare.

---

## Scenario 5: Everything Is Bad

```
WD = 0.89  (terrible)
ARI = 0.05  (essentially random)
NMI = 0.10
F1 = 0.03
UMAP is noise
```

**Diagnosis:** The generator has failed to converge. Likely causes:
- GAN training diverged (check training logs: G_loss → ∞ or oscillating).
- The dataset is too large for the model capacity (increase hidden dimensions).
- The learning rates are poorly tuned.

**Fix:**
- Check training logs first — identify which stage failed.
- If GAT loss > 0.5 after 7500 epochs → reduce `gat_epochs`, the model is underfitting.
- If D_loss → 0 while G_loss → ∞ → discriminator is too strong; reduce `nd_steps` or `d_lr`.
- Try training on a smaller dataset (Yan) first to verify the pipeline works.
- Reset `config.py` to `GARAGE_DEFAULTS` and try again.

---

## Quick Diagnosis Table

| WD | ARI | macro-F1 | Likely Issue | Action |
|----|-----|----------|-------------|--------|
| ✓ | ✓ | ✓ | Everything fine | Proceed. |
| ✓ | ✗ | ✗ | No cluster structure | Reduce leakage. |
| ✗ | ✓ | ✓ | Distributional shift | Re-normalise. |
| ✓ | ✓ | ✗ | Rare cells lost | Increase priority_weight. |
| ✗ | ✗ | ✗ | Training failure | Check losses, re-tune. |

---

## Related

- [Clustering Validation](howto/clustering_validation) — how each metric is computed.
- [Evaluation Metrics](background/evaluation_metrics) — mathematical definitions.
- [Troubleshooting](troubleshooting) — common error fixes.
