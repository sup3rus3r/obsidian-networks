---
name: architect-audio
description: Mutation strategy for audio domain architectures (spectrograms, waveforms, audio classification). Load when proposing mutations for audio tasks.
agent: architect
version: 1.0
domains: [audio]
---

# Architect Skill — Audio Domain

## Purpose

Audio architectures typically process either raw waveforms or time-frequency representations
(spectrograms, mel-filterbanks). The two paradigms have different constraints: waveform
models need very large receptive fields (audio at 16kHz means 16,000 samples per second),
while spectrogram models are essentially 2D image models with frequency as one axis and
time as the other — but frequency bins are not translationally equivalent like image pixels.
This skill guides the Architect toward mutations appropriate to audio's unique structure.

## Key Principles

- On synthetic data, audio is typically generated as a 1D waveform or a 2D spectrogram.
  The mutation strategy depends on which representation the base template uses.
- For waveform models, receptive field size is critical. A 1D CNN with kernel size 3
  and 5 layers only sees 11 samples — far too short for any audio pattern. Use dilated
  convolutions with large receptive fields.
- For spectrogram models, treat frequency and time axes differently — pooling along the
  frequency axis destroys harmonic structure; prefer pooling along the time axis first.
- Multi-scale temporal modelling is as important in audio as in timeseries.
- Prefer `free_form` (novel spectral or waveform operations) and `architecture_crossover`.

## Domain-Specific Guidance

### Approved Base Templates

- **CNN on spectrograms** — Treat mel-spectrogram as a 2D image with shape (time, freq, 1).
  Apply 2D convolutions, but be careful about frequency-axis pooling.
- **LSTM on sequences** — For waveform or frame-level feature sequences. Works best
  on MFCC-like features extracted per frame rather than raw waveform.

### Known-Good Mutation Combinations

- Frequency-Aware 2D Convolution (free_form) — Use 2D convolutions but with asymmetric
  kernels: tall in frequency (e.g. 3×1) to capture harmonic structure, wide in time
  (e.g. 1×5) to capture temporal patterns. Apply separately and fuse.
- Learnable Filterbank (free_form) — Replace hand-crafted mel-filterbank with a
  learnable 1D convolutional filterbank applied to raw waveforms. Each filter learns
  a frequency band end-to-end. Sinc-net-inspired but not identical.
- `architecture_crossover` with `fourier_neural_operator` — Apply Fourier transform
  to the time dimension of spectrograms, filter in frequency space, then IFFT back.
  Captures periodic audio patterns with global receptive field.
- Temporal Convolution + Attention (free_form) — Stack dilated causal convolutions
  with dilation [1, 2, 4, 8] to build a large temporal receptive field, then apply
  self-attention over the resulting high-level representations.
- Multi-scale Spectrogram Fusion (free_form) — Process the audio with multiple STFT
  window sizes (e.g. 256, 512, 1024 samples), generating spectrograms at different
  time-frequency resolutions. Fuse via learned weights before classification.

### Known Dead Ends

- Raw waveform convolutions with kernel size ≤ 7 — on 16kHz audio, kernel_size=7
  covers 0.44ms. No meaningful acoustic pattern exists at that resolution. Minimum
  useful kernel size for waveforms is 25–80 samples.
- Transposing frequency and time axes and treating them equivalently — frequency bins
  are not translationally equivalent. A pattern at 1kHz does not repeat at 2kHz.
- Global average pooling along the frequency axis before time aggregation — destroys
  harmonic structure before the model can learn it.
- Standard ViT applied to spectrograms without frequency-aware patching — uniform
  16×16 patches cut across harmonics. Patches should be tall (full frequency range)
  and narrow (short time windows).

### Common Failure Modes

- **Receptive field too small for waveform models** — A 5-layer 1D CNN with kernel=3
  has effective receptive field of only 11 samples. At 16kHz this is 0.7ms — far less
  than any speech phoneme (~50ms). Use dilated convolutions.
- **Spectrogram shape mismatch** — Mel-spectrogram shape is (n_mels, time_frames, 1)
  but some 2D Conv layers expect (height, width, channels) in a specific order. Check
  axis ordering explicitly in the code.
- **STFT window-hop mismatch** — If hop_length > win_length, spectrograms have gaps.
  Standard: hop_length = win_length // 4.
- **Class imbalance in synthetic audio** — Randomly generated audio classes must be
  balanced. If generating synthetic classes via amplitude or frequency variation,
  ensure equal samples per class.

## Examples

### Good Mutation Proposal
```json
{
  "architecture_name": "spectrogram_asymmetric_harmonic_cnn",
  "mutations": ["free_form", "attention_variant"],
  "rationale": "Use asymmetric 2D convolutions to separately capture harmonic (vertical frequency) and temporal (horizontal time) patterns in spectrograms, then fuse with temporal self-attention — tests whether explicit decomposition of audio structure improves classification over isotropic convolutions.",
  "free_form_description": "HarmonicTemporalCNN: apply two parallel Conv2D branches — one with kernel (n_mels//4, 1) to capture frequency structure across a frequency range, one with kernel (1, 7) to capture temporal patterns. Concatenate outputs channel-wise. Apply 2D self-attention over the time dimension only (attend over time steps, not frequency bins). Aggregate via time-axis pooling, then Dense classifier."
}
```

### Bad Mutation Proposal
```json
{
  "architecture_name": "lstm_deeper_audio",
  "mutations": ["depth_change"],
  "rationale": "More LSTM layers capture more complex audio patterns."
}
```
This is bad because: stacking LSTM layers without addressing receptive field, sampling
rate, or frequency structure is not an audio-specific mutation — it applies to any
sequential model. Low novelty.
