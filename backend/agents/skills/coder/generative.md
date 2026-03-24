---
name: coder-generative
description: Keras/TF implementation patterns for generative domain architectures (VAE, GAN). Load when generating code for generative tasks.
agent: coder
version: 1.0
domains: [generative]
---

# Coder Skill — Generative Domain

## Purpose

Generative model code fails most often due to: NaN losses in GANs (discriminator
saturation), VAE posterior collapse (KL overwhelming reconstruction), and training loop
errors when two networks (generator + discriminator or encoder + decoder) must be trained
with different optimisers. This skill provides correct patterns for VAE and GAN training loops.

## Key Principles

- GANs require two separate optimisers and alternating update steps. Do NOT use model.fit()
  for GAN training — use a custom training loop inside a `tf.keras.Model` with custom `train_step`.
- VAEs require a custom loss combining reconstruction + KL. Use the `add_loss()` API or
  override `train_step`.
- GAN label smoothing: real labels = 0.9 (not 1.0) to prevent discriminator saturation.
- VAE KL weight: start with β = 0.0 and anneal to 1.0 over training to prevent posterior collapse.
- Input shape for synthetic data: (batch, 64) flat vectors or (batch, 32, 32, 1) images.

## Synthetic Data Pattern

```python
np.random.seed(42)
# Flat vector synthetic data for simple VAE/GAN
X = np.random.randn(2000, 64).astype(np.float32)
# Normalise to [-1, 1] range (important for GAN tanh output)
X = X / (np.abs(X).max() + 1e-8)
```

## Correct Custom Training Patterns

### VAE with Correct ELBO Loss
```python
class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(latent_dim * 2),  # mean and log_var
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='tanh'),
        ])

    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return mean, log_var

    def reparameterise(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    def call(self, inputs, training=None):
        mean, log_var = self.encode(inputs)
        z = self.reparameterise(mean, log_var)
        recon = self.decoder(z)
        # Add KL loss to the model
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        self.add_loss(kl_loss)
        return recon

# Compile with reconstruction loss only; KL is added via add_loss
vae = VAE(input_dim=64, latent_dim=16)
vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
vae.fit(X, X, epochs=30, batch_size=64, validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)])
vae.save('output/model.keras')
```

### GAN with Custom train_step
```python
class GAN(tf.keras.Model):
    def __init__(self, latent_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='tanh'),
        ])
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1),  # logit (no sigmoid — use from_logits=True)
        ])

    def compile(self, g_optimizer, d_optimizer, **kwargs):
        super().compile(**kwargs)
        self.g_opt = g_optimizer
        self.d_opt = d_optimizer
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as d_tape:
            fake = self.generator(noise, training=True)
            real_logits = self.discriminator(real_data, training=True)
            fake_logits = self.discriminator(fake, training=True)
            # Label smoothing: real=0.9, fake=0.0
            d_loss = (
                self.loss_fn(tf.ones_like(real_logits) * 0.9, real_logits) +
                self.loss_fn(tf.zeros_like(fake_logits), fake_logits)
            )
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            fake = self.generator(noise, training=True)
            fake_logits = self.discriminator(fake, training=False)
            g_loss = self.loss_fn(tf.ones_like(fake_logits), fake_logits)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def call(self, inputs, training=None):
        # For saving: the full model call is the generator
        return self.generator(inputs, training=training)

gan = GAN(latent_dim=16, output_dim=64)
gan.compile(
    g_optimizer=tf.keras.optimizers.Adam(1e-4),
    d_optimizer=tf.keras.optimizers.Adam(1e-4),
)
gan.fit(X, epochs=50, batch_size=64)
# Save only the generator for evaluation
gan.generator.save('output/model.keras')
```

## Common Implementation Errors

- **GAN using model.fit() directly** — Standard fit() passes (X, y) pairs, but GAN training
  needs real data only (no labels) and alternating optimiser steps. Use custom `train_step`.
- **Sigmoid + from_logits=True conflict** — If the discriminator output has `activation='sigmoid'`,
  do NOT use `from_logits=True` in the loss. Use one or the other, not both.
- **VAE KL collapse** — If KL loss is very large initially, the decoder ignores the latent.
  Fix: multiply KL by a small weight (β=0.001) and anneal toward 1.0.
- **NaN in GAN loss** — Occurs when discriminator outputs saturate. Fix: remove sigmoid
  from discriminator, use `from_logits=True` in loss, and apply label smoothing (0.9/0.0).

## Output Configuration

- VAE: save `vae.save('output/model.keras')` (full VAE) or decoder only for generation
- GAN: save `gan.generator.save('output/model.keras')` — generator is the inference model
