---
name: coder-audio
description: Keras/TF implementation patterns for audio domain architectures. Load when generating code for audio tasks.
agent: coder
version: 1.0
domains: [audio]
---

# Coder Skill — Audio Domain

## Purpose

Audio code fails most often due to: incorrect 1D vs 2D convolution choices, missing
causal padding in temporal models, and tf.signal operations not being in the allowed
imports. All audio processing must use numpy or basic tensorflow ops — no librosa,
no scipy.signal, no torchaudio. Spectrograms must be computed manually using tf.signal.stft
or simulated as random 2D arrays for synthetic data.

## Key Principles

- Allowed audio ops: `tf.signal.stft`, `tf.signal.rfft`, `tf.signal.fft`. No librosa.
- For synthetic data, generate random mel-spectrogram-shaped arrays: (batch, time_frames, n_mels, 1).
- For waveform models, input shape is (batch, n_samples, 1). Use Conv1D with causal padding.
- Do not use padding='same' in temporal Conv1D for audio — use padding='causal' or explicit padding.
- Audio classification output: Dense(n_classes, softmax) after GlobalAveragePooling over time.

## Synthetic Data Pattern

```python
np.random.seed(42)
# Simulate mel-spectrogram features: (N, time_frames, n_mels, 1)
n_samples, time_frames, n_mels, n_classes = 1000, 64, 40, 5
X = np.random.randn(n_samples, time_frames, n_mels, 1).astype(np.float32)
# Normalise to approximate mel-spectrogram scale
X = (X - X.mean()) / (X.std() + 1e-8)
y = np.random.randint(0, n_classes, n_samples).astype(np.int32)
```

## Correct Custom Layer Patterns

### Asymmetric Frequency-Time Conv2D
```python
class FreqTimeConv2D(tf.keras.layers.Layer):
    """Separate convolutions along frequency and time axes."""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        # Frequency conv: tall kernel (10 bins × 1 frame)
        self.freq_conv = tf.keras.layers.Conv2D(
            filters // 2, (10, 1), padding='same', use_bias=False
        )
        # Time conv: wide kernel (1 bin × 5 frames)
        self.time_conv = tf.keras.layers.Conv2D(
            filters // 2, (1, 5), padding='same', use_bias=False
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        freq_out = self.freq_conv(inputs)  # (batch, T, F//2, filters//2)
        time_out = self.time_conv(inputs)  # (batch, T, F, filters//2) — different shape
        # Project freq_out to same freq dim as time_out via average pooling
        freq_out = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(freq_out)
        freq_out = tf.tile(freq_out, [1, tf.shape(inputs)[1], tf.shape(inputs)[2] // freq_out.shape[2], 1])
        out = tf.concat([freq_out, time_out], axis=-1)
        return self.bn(tf.nn.relu(out), training=training)
```

### Learnable 1D Filterbank (waveform model)
```python
class LearnableFilterbank(tf.keras.layers.Layer):
    """Replace mel-filterbank with learned Conv1D filters on waveform."""
    def __init__(self, n_filters, filter_length, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(
            n_filters, filter_length,
            strides=filter_length // 4,  # 75% overlap
            padding='same',
            use_bias=False,
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        # inputs: (batch, n_samples, 1)
        out = self.conv(inputs)              # (batch, time_frames, n_filters)
        out = tf.abs(out)                    # rectify (like power spectrogram)
        return self.bn(out, training=training)
```

## Common Implementation Errors

- **librosa or scipy imported** — These are not in the allowed import list. All spectrogram
  computation must use tf.signal.stft or pre-computed synthetic numpy arrays.
- **Wrong Conv1D padding for temporal audio** — `padding='same'` in Conv1D looks ahead in time.
  For causal waveform models use `padding='causal'`. For spectrogram (non-causal) use `padding='same'`.
- **2D vs 1D mismatch** — Spectrogram data is 2D (time × frequency), requiring Conv2D.
  Waveform data is 1D (time), requiring Conv1D. Do not apply Conv2D to waveform data.
- **Frequency axis pooled too early** — `MaxPool2D((2, 2))` reduces both time and frequency.
  For audio, pool time aggressively and frequency conservatively: `MaxPool2D((2, 1))`.

## Output Configuration

- Audio classification: `GlobalAveragePooling2D()` (over time and freq) + `Dense(n_classes, softmax)` + `sparse_categorical_crossentropy`
- Sequential audio (frame-by-frame): `TimeDistributed(Dense(n_classes, softmax))` — one prediction per frame
