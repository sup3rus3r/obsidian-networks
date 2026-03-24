---
name: coder-timeseries
description: Keras/TF implementation patterns for timeseries domain architectures. Load when generating code for timeseries/forecasting tasks.
agent: coder
version: 1.0
domains: [timeseries]
---

# Coder Skill — Timeseries Domain

## Purpose

Timeseries code fails most often due to: causal padding mistakes (allowing future data
leakage), wrong output shape for multi-step forecasting, and LSTM return_sequences
mismatches in stacked layers. This skill provides correct patterns for temporal
architectures and flags the errors that produce numerically correct but causally invalid models.

## Key Principles

- For forecasting, all Conv1D must use causal padding: `padding='causal'`. Never use
  `padding='same'` or `padding='valid'` in a forecasting model.
- Stacked LSTMs: all layers except the last must have `return_sequences=True`.
  The last LSTM should have `return_sequences=False` unless temporal attention follows.
- For multi-step forecasting with horizon H, the output Dense must have `units=H`.
  Do not output 1 unit and repeat.
- Input shape is (batch, seq_len, n_features) = (batch, 50, 1) for synthetic data.

## Synthetic Data Pattern

```python
np.random.seed(42)
# Forecasting: sine wave with noise
t = np.linspace(0, 100, 5000)
signal = np.sin(t) + 0.1 * np.random.randn(5000)
seq_len, horizon = 50, 10
X = np.array([signal[i:i+seq_len] for i in range(len(signal)-seq_len-horizon)]).astype(np.float32)
y = np.array([signal[i+seq_len:i+seq_len+horizon] for i in range(len(signal)-seq_len-horizon)]).astype(np.float32)
X = X[..., np.newaxis]  # (N, 50, 1)
```

## Correct Custom Layer Patterns

### Dilated Causal Convolution Block
```python
class DilatedCausalConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(
            filters, kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',   # MUST be causal for forecasting
            activation='relu',
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        return self.norm(x, training=training)
```

### Temporal Attention over LSTM States
```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W = tf.keras.layers.Dense(units, use_bias=False)
        self.v = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, hidden_states, training=None):
        # hidden_states: (batch, seq_len, units)
        score = self.v(tf.nn.tanh(self.W(hidden_states)))  # (batch, seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)             # (batch, seq_len, 1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # (batch, units)
        return context
```

### Instance Normalization (per-sequence)
```python
class InstanceNorm1D(tf.keras.layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, inputs, training=None):
        # inputs: (batch, seq_len, features)
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        std  = tf.math.reduce_std(inputs, axis=1, keepdims=True)
        return (inputs - mean) / (std + self.eps)
```

## Common Implementation Errors

- **Causal padding omitted** — Using `Conv1D(..., padding='same')` in a forecasting model
  allows the model to see future timesteps during training. Use `padding='causal'` always.
- **Wrong output shape** — For horizon=10, the output must be `Dense(10, activation='linear')`.
  Using `Dense(1)` and tiling produces a flat forecast, not a learned multi-step output.
- **LSTM return_sequences mismatch** — In a 2-layer stacked LSTM: layer 1 must have
  `return_sequences=True`, layer 2 must have `return_sequences=False` (unless followed
  by temporal attention, in which case layer 2 also needs `return_sequences=True`).
- **Non-stationary data without normalisation** — Without per-sequence normalisation,
  a sine wave with amplitude 10 and one with amplitude 0.1 in the same batch cause
  gradient imbalance. Always normalise input per-sequence.

## Output Configuration

- Forecasting: `Dense(horizon, activation='linear')` + `mse` loss + `mae` metric
- Classification of sequences: `Dense(n_classes, activation='softmax')` + `sparse_categorical_crossentropy`
- Anomaly detection (reconstruction): output shape = input shape; loss = `mse`
