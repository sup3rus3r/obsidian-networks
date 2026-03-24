---
name: researcher-generative
description: arXiv query strategy and paper selection guidance for generative domain research. Load when running research for generative model tasks.
agent: researcher
version: 1.0
domains: [generative]
---

# Researcher Skill — Generative Domain

## Purpose

Generative model research is dominated by large-scale diffusion and LLM papers that are
impossible to implement in a self-contained training script. The most useful papers for
this pipeline describe novel latent space structures, loss formulations, or training
dynamics for VAEs and GANs — models that can be trained from scratch on synthetic data
in minutes. Avoid targeting diffusion transformer or score-based model papers that require
pre-training at scale.

## Key Principles

- Target VAE and GAN architectural innovations, not diffusion or flow-matching papers
  (those require large-scale pre-training).
- The most extractable mechanisms are: novel prior distributions, novel posterior
  approximations, GAN regularisation terms, and latent space structure innovations.
- Avoid papers where "generative" means "large language model" — target image/data
  generation architecture papers.

## Procedure

1. For each slot, generate one query targeting VAE latent structure innovations and one
   targeting GAN training stability or discriminator design.
2. Select papers with explicit loss formulation (VAE ELBO variant, GAN objective variant).
3. Extract: the modified ELBO term, the discriminator regularisation formula, or the
   latent structure computation.

## High-Value Query Angles for Generative

- VAE latent structure: `"variational autoencoder hierarchical latent space architecture novel"`
- Vector quantisation: `"vector quantized variational autoencoder VQ-VAE discrete latent"`
- GAN stability: `"GAN training stability spectral normalization gradient penalty discriminator"`
- Normalising flows: `"normalizing flow prior variational autoencoder expressive distribution"`
- Beta-VAE / disentanglement: `"disentangled representation variational autoencoder beta mechanism"`
- Conditional generation: `"conditional VAE GAN architecture generation mechanism"`
- Energy-based: `"energy-based model generative training contrastive divergence"`
- Adversarial training objectives: `"Wasserstein GAN training objective divergence measure"`

## Paper Selection Criteria

**Select if abstract contains:**
- Modified VAE objective (ELBO variant, KL weighting, free bits)
- Novel GAN regularisation (spectral norm, gradient penalty, R1)
- Latent space structure (hierarchical, discrete, flow-based prior)
- GAN discriminator architecture innovation

**Skip if abstract primarily contains:**
- Diffusion models, score matching, or DDPM variants
- Large-scale generative models (requires pre-training at scale)
- Text-to-image generation papers
- Only FID/IS benchmark comparisons without architectural description

## Mechanism Extraction Focus

- The modified ELBO: exact KL weighting or free bits formulation
- The discriminator regularisation: gradient penalty formula, spectral norm application
- The latent structure: hierarchical prior formula, VQ codebook update rule
- Any conditional generation mechanism: how class/label conditions the generation
