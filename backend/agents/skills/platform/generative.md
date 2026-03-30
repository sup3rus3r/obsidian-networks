---
name: platform-generative
description: Guides the platform agent through RESEARCH → PLAN → BUILD for generative models (VAE, GAN, diffusion). Load when the user describes a generative modelling task.
agent: platform
version: 1.0
domains: [generative]
---

# Platform Skill — Generative Domain

## Purpose

Generative models have non-standard training loops: VAEs require the ELBO loss (reconstruction
+ KL divergence), GANs require alternating generator/discriminator updates, and neither fits
into `model.fit()` cleanly. This skill guides the agent in researching the specific generative
architecture, planning the loss function and training loop precisely, and generating code that
uses `tf.GradientTape` for custom training rather than `model.fit()`.

## Key Principles

- VAE training loop: forward pass (encoder → reparameterize → decoder) → ELBO loss → backprop. NEVER use `model.fit()` for VAE.
- GAN training loop: ALTERNATE discriminator step and generator step each batch. NEVER update both in one `model.fit()` call.
- Reparameterization trick: `z = mu + epsilon * sigma` where `epsilon = tf.random.normal(shape=tf.shape(mu))`.
- Save the primary model (`encoder` for VAE, `generator` for GAN) to `output/model.keras`.
- KL loss weight (β) should be annealed from 0 to 1 during training to prevent posterior collapse.
- All image data in scripts must be float32 normalized to [0, 1] (VAE/DCGAN) or [-1, 1] (DCGAN with tanh output).

## Procedure

### RESEARCH Phase

**CRITICAL: Never ingest paper URLs from your training knowledge. Only ingest URLs returned by arXiv search results — do not use papers you already know.**

1. Run both arXiv searches using the `search_arxiv` tool — use only the returned results:
   - `"variational autoencoder VAE ELBO loss reparameterization image 2024"`
   - `"GAN generative adversarial network training stability DCGAN 2023 2024"`
2. Select 3–4 papers. For VAE: focus on KL annealing, β-VAE. For GAN: focus on DCGAN, training stability, gradient penalty.
3. Ingest in parallel.
4. Fetch TF/Keras docs in parallel:
   - `"GradientTape custom training loop TensorFlow generative"`
   - `"Conv2DTranspose upsampling decoder architecture"`
   - `"BatchNormalization generative model training"`
5. Call `finalize_research()`.

### PLANNING Phase

Query at least 6 times before writing:
- `"VAE encoder decoder architecture latent dimension image"`
- `"KL divergence annealing beta-VAE training stability"`
- `"DCGAN generator discriminator architecture filter sizes"`
- `"GAN training instability mode collapse solutions"`
- `"Conv2DTranspose stride upsampling decoder image generation"`
- `"latent dimension size VAE GAN image quality"`

The plan MUST include:
- Architecture type: VAE or GAN (with clear justification from research)
- `latent_dim` (e.g. 64–256) with source
- Encoder architecture: Conv2D blocks with strides, flatten, Dense(latent_dim × 2 for VAE)
- Decoder architecture: Dense → Reshape → Conv2DTranspose blocks to output shape
- Exact loss function: ELBO for VAE, binary_crossentropy + optional gradient penalty for GAN
- Training loop description: custom `tf.GradientTape` loop, NOT `model.fit()`
- Data normalization: [0, 1] for sigmoid output (VAE), [-1, 1] for tanh output (GAN)

### BUILD Phase

Before writing any code, call `query_research` for each key implementation detail (layer sizes, learning rates, loss functions, hyperparameters) to retrieve exact values from the ingested papers. Every value in the script must match the approved plan — `create_notebook` runs an automated alignment check and will reject mismatches.

Follow the standard BUILD SEQUENCE. Key generative-specific steps:

1. In `edit_script`, always use a custom training loop:
   ```python
   for epoch in range(epochs):
       for batch in dataset:
           with tf.GradientTape() as tape:
               # VAE: forward pass + ELBO loss
               # GAN: separate tape for generator and discriminator
           grads = tape.gradient(loss, model.trainable_variables)
           optimizer.apply_gradients(zip(grads, model.trainable_variables))
   ```
2. For VAE: implement reparameterization as a separate function or `layers.Lambda`.
3. For GAN: use TWO optimizers (one per model), TWO `GradientTape` contexts per batch.
4. Synthetic data for description mode: `X = np.random.uniform(0, 1, (500, 32, 32, 3)).astype(np.float32)`

## Domain-Specific Guidance

### VAE Architecture Blueprint

```
Encoder: Input(H,W,C) → Conv2D(32,3,s=2)→BN→ReLU → Conv2D(64,3,s=2)→BN→ReLU → Flatten → Dense(256)→ReLU
         → Dense(latent_dim) [mu], Dense(latent_dim) [log_var]
Reparameterize: z = mu + exp(0.5*log_var) * N(0,1)
Decoder: Dense(8*8*64)→ReLU → Reshape(8,8,64) → ConvT(64,3,s=2)→BN→ReLU → ConvT(32,3,s=2)→BN→ReLU
         → ConvT(C, 3, s=1, activation='sigmoid')
Loss: reconstruction_loss = mean(binary_crossentropy(x_flat, x_pred_flat)) * H*W*C
      kl_loss = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
      total = reconstruction_loss + beta * kl_loss
```

### GAN Architecture Blueprint

```
Generator: Input(latent_dim,) → Dense(8*8*128,relu) → Reshape(8,8,128) → ConvT(64,4,s=2)→BN→ReLU
           → ConvT(32,4,s=2)→BN→ReLU → ConvT(C,4,s=1,activation='tanh')
Discriminator: Input(H,W,C) → Conv2D(64,4,s=2,leaky_relu) → Conv2D(128,4,s=2,leaky_relu) → Flatten → Dense(1,sigmoid)
Loss: D: binary_crossentropy(real_labels=1, fake_labels=0) + binary_crossentropy(fake_from_G, 0)
      G: binary_crossentropy(fake_from_G, 1) [fool the discriminator]
```

### Known Dead Ends

- `model.fit()` for GAN — alternating updates are impossible with a single `fit()` call.
- Sigmoid output for GAN generator — use `tanh` so outputs are in [-1, 1]; discriminator receives [-1, 1] range real images too.
- No KL annealing in VAE — posterior collapse occurs; start β at 0 and linearly increase to 1.
- Very small `latent_dim` (< 16) for image VAE — decoder cannot reconstruct detail; use ≥ 64.
- Dense decoder without Conv2DTranspose — produces blurry reconstructions on images; always upsample spatially.

### Common Failure Modes

- **Mode collapse in GAN** — discriminator converges before generator trains meaningfully. Add gradient penalty or use label smoothing (real labels = 0.9, fake labels = 0.1).
- **KL loss dominates** — reconstruction loss becomes 0 and KL drives all gradients. Use β-annealing: β starts at 0.0 and reaches 1.0 at epoch 10.
- **Conv2DTranspose checkerboard artifacts** — caused by kernel size not divisible by stride. Use stride=2, kernel=4 (DCGAN standard).
- **Shape mismatch in decoder reshape** — Dense output size must exactly equal the reshape target: `Dense(h * w * c)` → `Reshape((h, w, c))`. Verify arithmetic.

## Examples

### Good Plan Training Loop Description

```
## 5. Training Strategy
Custom tf.GradientTape loop (no model.fit):
- Per batch: encoder_tape → z = reparameterize(mu, log_var) → decoder → ELBO loss → backprop through encoder+decoder
- β schedule: β = min(1.0, epoch / 10.0) — KL weight annealed over first 10 epochs  [Higgins 2017, https://...]
- Optimizer: Adam(lr=2e-4)
- Epochs: 50 (EarlyStopping on val reconstruction_loss, patience=10)
- Save encoder to output/model.keras
```

### Bad Plan Training Loop Description

```
## 5. Training Strategy
model.fit(X_train, X_train, epochs=50)
```
This is bad: `model.fit()` cannot implement ELBO loss or alternating GAN updates.
