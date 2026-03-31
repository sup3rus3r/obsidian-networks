---
name: platform-audio
description: Guides the platform agent through RESEARCH → PLAN → BUILD for audio classification tasks (CNN on spectrograms, conformer-style architectures). Load when the user describes an audio or speech classification task.
agent: platform
version: 1.0
domains: [audio]
---

# Platform Skill — Audio Domain

## Purpose

Audio models on the platform operate on pre-computed mel-spectrogram representations
(2D arrays of shape `(n_mels, time_frames, 1)`), not raw waveforms. The agent must plan
a Conv2D-based or conformer-style architecture that treats spectrograms as images with
time and frequency axes. This skill guides the agent in selecting the right research papers,
planning the spectrogram shape and model architecture, and generating code that adds the
correct channel dimension and uses GlobalAveragePooling2D or temporal pooling.

## Key Principles

- Spectrogram input shape is `(n_mels, time_frames, 1)` — the channel dimension must always be present.
- In description mode (no dataset uploaded), ALWAYS call `fetch_tensorflow_datasets` first to find a suitable public audio dataset (e.g. speech_commands, groove/full-midionly, fuss). If one is found, write `tfds.load(dataset_name, split="train", as_supervised=True)` directly into the training script — the dataset downloads on the user's machine when they run the notebook. Do NOT download it on the server, do NOT use `dataset.csv` for tfds data. Only fall back to `np.random.uniform(0, 1, (1000, 64, 32, 1)).astype(np.float32)` synthetic spectrograms if no suitable dataset exists.
- Do NOT use 1D convolutions on raw waveforms — the platform pipeline delivers spectrograms.
- For conformer-style: subsampling Conv2D → Reshape → Transformer blocks → GlobalAveragePooling1D → Dense.
- For CNN-audio: Conv2D stacks along frequency and time axes → GlobalAveragePooling2D → Dense.
- Output is always classification: `Dense(n_classes, softmax)` with `sparse_categorical_crossentropy`.

## Procedure

### RESEARCH Phase

**CRITICAL: Never ingest paper URLs from your training knowledge. Only ingest URLs returned by arXiv search results — do not use papers you already know.**

1. Run both arXiv searches using the `search_arxiv` tool — use only the returned results:
   - `"audio classification mel spectrogram CNN deep learning 2024"`
   - `"conformer speech recognition spectrogram transformer audio 2023 2024"`
2. Select 3–4 papers. Focus on: spectrogram input shape, Conv2D filter sizes, conformer block design.
3. Ingest in parallel.
4. Fetch TF/Keras docs in parallel:
   - `"Conv2D GlobalAveragePooling2D image classification audio spectrogram"`
   - `"MultiHeadAttention conformer audio transformer block"`
   - `"sparse_categorical_crossentropy accuracy audio classification"`
5. Call `finalize_research()`.

### PLANNING Phase

Query at least 6 times before writing:
- `"mel spectrogram frequency bins time frames audio classification"`
- `"CNN architecture audio spectrogram filter sizes kernel"`
- `"conformer block convolution attention hybrid audio speech"`
- `"batch normalization audio spectrogram training stability"`
- `"data augmentation SpecAugment frequency time masking audio"`
- `"learning rate schedule audio classification training"`

The plan MUST include:
- Spectrogram shape: `(n_mels, time_frames, 1)` — default `(64, 32, 1)` if unspecified
- Architecture type: CNN-audio or conformer — justified by dataset size and task complexity
- For CNN: exact Conv2D filter counts, kernel sizes, pooling strategy
- For conformer: subsampling block config, number of conformer blocks, attention heads
- Number of output classes
- Augmentation: SpecAugment frequency/time masking if dataset is small

### BUILD Phase

Before writing any code, call `query_research` for each key implementation detail (layer sizes, learning rates, loss functions, hyperparameters) to retrieve exact values from the ingested papers. Every value in the script must match the approved plan — `create_notebook` runs an automated alignment check and will reject mismatches.

Follow the standard BUILD SEQUENCE. Key audio-specific steps:

1. In `edit_script`:
   - Always add channel dim: `if X.ndim == 3: X = X[..., np.newaxis]`
   - `Input(shape=(n_mels, time_frames, 1))` — match the actual spectrogram shape
   - For CNN: Conv2D(32,3,relu)→BN → Conv2D(64,3,relu)→BN → MaxPool2D(2) → Conv2D(128,3,relu)→BN → GlobalAveragePooling2D() → Dense(128,relu) → Dense(n_classes,softmax)
   - For conformer: Conv2D(128,3,stride=2)→Reshape→TransformerBlocks→GlobalAveragePooling1D→Dense
2. Compile with `loss='sparse_categorical_crossentropy', metrics=['accuracy']`
3. For description mode with tfds dataset: use `tfds.load(dataset_name, split="train", as_supervised=True)` — pipeline with `.map()` to extract/reshape spectrograms, `.batch()`, `.prefetch()`. No `dataset.csv`.
   For description mode with no suitable dataset (synthetic fallback): `X = np.random.uniform(0,1,(1000,64,32,1)).astype(np.float32); y = np.random.randint(0,10,(1000,))`

## Domain-Specific Guidance

### Approved Architecture Patterns

- **CNN-audio** (3–4 Conv2D blocks, 32→64→128 filters, 3×3 kernels, MaxPool2D after block 2) — reliable baseline for sound classification with ≥ 500 samples
- **Conformer-Lite** (subsampling Conv2D → 4 conformer blocks, 4 attention heads, embed_dim=128) — for speech recognition or complex temporal patterns

### Spectrogram Dimension Guide

| n_mels | Frequency resolution | Use case |
|---|---|---|
| 40 | Low | Speech phoneme classification |
| 64 | Medium (default) | Environmental sound classification |
| 128 | High | Music/instrument classification |

| time_frames | Duration | Notes |
|---|---|---|
| 32 | ~0.5s | Short sounds, keyword spotting |
| 64–128 | 1–2s | Most audio classification tasks |

### Known Dead Ends

- Raw waveform 1D Conv — platform delivers spectrograms; 1D waveform models require a different preprocessing pipeline.
- Too many MaxPool2D layers on small spectrograms — a (64,32) input with 3 MaxPool2D(2) reduces to (8,4); add no more than 2.
- GlobalAveragePooling2D before all spatial reduction — averages over too large a map; use only after ≥ 2 Conv+Pool blocks.
- SpecAugment applied to test set — augmentation must be training-time only.
- Missing channel dimension in Input — `Input(shape=(n_mels, time_frames))` causes Conv2D to fail; always include the channel: `(n_mels, time_frames, 1)`.

### Common Failure Modes

- **Channel dimension error** — `X.shape` is `(N, n_mels, time_frames)` without a channel axis. Always run `X = X[..., np.newaxis]` at the start of the script.
- **MaxPool2D collapsing time axis** — spectrogram is taller than wide (64×32). MaxPool2D(2) halves both dimensions equally; after 2 pools, time axis is only 8 steps. Stop pooling before time axis < 4.
- **Conformer Reshape error** — subsampling Conv2D changes both spatial dims. Compute the output shape explicitly: `h_out = ceil(n_mels/2), t_out = ceil(time_frames/2)`. Then `Reshape((h_out * t_out, 128))` before transformer blocks.
- **Accuracy does not improve** — often caused by class imbalance in audio datasets. Add `class_weight` if any class has < 10% of samples.

## Examples

### Good Plan Architecture Section

```
### Selected Model: CNN-Audio Classifier
Input: (64, 32, 1) — 64 mel bins, 32 time frames, 1 channel  [Hershey 2017, https://...]
Conv2D(32, 3, relu, padding='same') → BatchNorm
Conv2D(64, 3, relu, padding='same') → BatchNorm → MaxPool2D(2)   → output: (32, 16, 64)
Conv2D(128, 3, relu, padding='same') → BatchNorm → MaxPool2D(2)  → output: (16, 8, 128)
GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.3) → Dense(10, softmax)
Loss: sparse_categorical_crossentropy. Optimizer: Adam(lr=1e-3). Metric: accuracy.
```

### Bad Plan Architecture Section

```
### Selected Model: Audio Classifier
Some convolutional layers on the audio data.
```
This is bad: no spectrogram shape, no filter counts, no pooling strategy, no class count, no citations.
