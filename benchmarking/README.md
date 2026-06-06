# Benchmarking

Baseline generative models for comparing against GARAGE.

## Layout

```
benchmarking/
├── sota/                         General-purpose generative baselines
│   ├── gan.py                    Vanilla GAN (BCE loss, Adam)
│   ├── wgan.py                   Wasserstein GAN (RMSprop, weight clipping)
│   ├── fgan.py                   f-divergence GAN (Fisher ratio + constraint)
│   ├── vae.py                    Variational Autoencoder (MSE + KL)
│   ├── lsh_gan.py                LSH-GAN (KNN subsample + GAN)
│   └── *_tf1.py                  Original TF1.11 reference implementations
└── scrna_seq_specific/           scRNA-seq-specific baselines
    ├── scgan.py                  scGAN (WGAN-GP, deep architecture)
    ├── scvae.py                  scVAE (beta-VAE, deep encoder/decoder)
    ├── scdiffusion.py            scDiffusion (DDPM, MLP denoiser)
    ├── gan_ros.py                GAN + Random Oversampling (ROS)
    └── vae_ros.py                VAE + Random Oversampling (ROS)
```

All PyTorch files accept `--dataset {yan,pollen,cbmc,muraro}` and save to `data/gen_data/<method>/`.
