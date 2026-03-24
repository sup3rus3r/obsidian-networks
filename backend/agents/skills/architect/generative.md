---
name: architect-generative
description: Mutation strategy for generative domain architectures (VAEs, GANs, diffusion-inspired models). Load when proposing mutations for generative tasks.
agent: architect
version: 1.0
domains: [generative]
---

# Architect Skill — Generative Domain

## Purpose

Generative models learn to produce new samples from a data distribution. The two dominant
paradigms — VAEs (encoder-decoder with latent space) and GANs (generator-discriminator
adversarial training) — have fundamentally different failure modes. VAEs produce blurry
samples but train stably; GANs produce sharp samples but suffer from mode collapse and
training instability. This skill guides the Architect toward mutations that address
these failure modes or combine the two paradigms in novel ways.

## Key Principles

- GAN training stability is the primary challenge — the discriminator must not outpace
  the generator. Any GAN mutation should include a mechanism to balance training dynamics
  (spectral normalization, gradient penalty, or discriminator step ratio).
- VAE mutations should address posterior collapse (the KL term dominating and the decoder
  ignoring the latent code). KL annealing, free bits, or β-VAE weighting address this.
- The latent space is where novelty lives. Mutations to latent structure (hierarchical
  latents, disentangled dimensions, flow-based prior) are higher value than mutations
  to the encoder/decoder architecture alone.
- On synthetic data (small, 32×32 or 1D arrays), very large discriminators/encoders
  overfit. Keep total params < 2M.
- Prefer `free_form` (novel latent mechanisms, novel loss formulations) and
  `architecture_crossover` (e.g. flow-based, energy-based, diffusion-inspired).

## Domain-Specific Guidance

### Approved Base Templates

- **VAE** — Stable training, useful latent space, blurry reconstruction. Mutate
  the prior, the posterior parameterisation, or the decoder architecture.
- **GAN** — Sharp generation, unstable training. Mutate the discriminator objective,
  the noise injection, or the generator conditioning.

### Known-Good Mutation Combinations

- Hierarchical Latent VAE (free_form) — Use multiple levels of latent variables
  (z1 global, z2 local). Top-down prior for each level conditions on higher-level z.
  NVAE-inspired. Tests whether hierarchical structure improves latent coverage.
- Spectral Normalisation + Gradient Penalty GAN (free_form) — Apply spectral norm
  to all discriminator weight matrices AND add an R1 gradient penalty to the discriminator
  loss. Combines two complementary stability mechanisms. Tests whether dual regularisation
  outperforms each alone.
- VQ-VAE (architecture_crossover with free_form) — Replace continuous latent with
  a discrete codebook. Encoder output is quantized to nearest codebook vector.
  Commitment loss + codebook loss added to reconstruction. Tests discrete vs continuous
  latent for representation quality.
- Flow-Based Prior VAE (architecture_crossover with `free_form`) — Replace the standard
  N(0,I) prior with a normalising flow (e.g. RealNVP) that learns a more expressive
  prior distribution. Tests whether expressive priors reduce posterior-prior mismatch.
- Diffusion Decoder (architecture_crossover) — Use the VAE encoder as usual but replace
  the decoder with a denoising network that iteratively refines a noisy sample toward
  the reconstruction. Tests diffusion-style generation within a VAE framework.

### Known Dead Ends

- Adding more Dense layers to GAN generator/discriminator without normalisation —
  unstable training, discriminator wins immediately.
- Very high-dimensional latent spaces (dim > 256) on small synthetic data — the model
  fails to fill the latent space; posterior collapses to the prior everywhere.
- Standard Batch Normalization in GAN discriminator — BN in the discriminator is
  problematic (correlates batch samples, breaking the per-sample discriminator assumption).
  Use Layer Normalization or Spectral Normalization instead.
- Skip connections from encoder to decoder in a GAN discriminator — this is a VAE
  pattern applied to a GAN, which creates a shortcut that bypasses the latent code.

### Common Failure Modes

- **GAN mode collapse** — Generator produces one or few samples regardless of noise input.
  Symptom: discriminator loss → 0 while generator loss → high. Fix: reduce discriminator
  learning rate, add noise to discriminator inputs, or use Wasserstein loss.
- **VAE posterior collapse** — KL term → 0, decoder ignores latent code and generates
  the mean of the data. Fix: KL annealing (start β=0, ramp to 1 over training),
  or free bits (KL clipped to minimum value per dimension).
- **NaN losses in GAN** — Occurs when discriminator outputs saturate (sigmoid → 0 or 1).
  Fix: label smoothing (real labels = 0.9, not 1.0), LeakyReLU in discriminator,
  clip discriminator weights (WGAN) or use gradient penalty (WGAN-GP).
- **Reconstruction loss vs KL balance** — If reconstruction loss (e.g. MSE) is in pixel
  space and KL is per-latent-dimension, they can be on very different scales. Always
  weight: `loss = reconstruction_loss + β * kl_loss` and start with β << 1.

## Examples

### Good Mutation Proposal
```json
{
  "architecture_name": "vae_vq_hierarchical_latent",
  "mutations": ["free_form", "architecture_crossover"],
  "rationale": "Combine vector quantization with a two-level hierarchical latent structure — tests whether discrete, multi-scale latent representations reduce posterior collapse while preserving generation diversity compared to a single continuous latent.",
  "free_form_description": "HierarchicalVQVAE: Encoder produces two latent maps at different spatial resolutions (bottom z1 = high-res detail, top z2 = low-res structure). Each is quantized against a separate codebook. Decoder receives concatenated upsampled z1 and z2. Loss = reconstruction + commitment_loss_1 + commitment_loss_2. No KL term needed (discrete latent has no posterior collapse)."
}
```

### Bad Mutation Proposal
```json
{
  "architecture_name": "gan_deeper_generator",
  "mutations": ["depth_change"],
  "rationale": "Deeper generator captures more complex distributions."
}
```
This is bad because: adding Dense layers to a GAN generator without addressing
training stability is the primary cause of mode collapse, not a solution. Low novelty,
high failure risk.
