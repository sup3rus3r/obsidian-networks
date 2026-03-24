---
name: researcher-audio
description: arXiv query strategy and paper selection guidance for audio domain research. Load when running research for audio tasks.
agent: researcher
version: 1.0
domains: [audio]
---

# Researcher Skill — Audio Domain

## Purpose

Audio architecture research sits at the intersection of signal processing and deep learning.
The most useful papers combine classical signal processing ideas (Fourier analysis, filterbanks,
spectral decomposition) with neural network operations. Papers that propose novel filterbank
designs, frequency-aware convolutions, or audio-specific attention formulations contain
the most extractable mechanisms.

## Key Principles

- The best audio architecture papers bridge signal processing theory and neural computation.
  Look for papers that replace hand-crafted processing steps with learned equivalents.
- Temporal multi-scale processing is the highest-value direction — different audio events
  occur at different timescales.
- Spectrogram-based models are closer to vision; waveform-based models are more novel.
  Prioritise waveform papers for higher novelty.

## Procedure

1. For each slot, generate one query targeting frequency-domain or filterbank operations
   and one targeting temporal attention or multi-scale audio representation.
2. Select papers describing learnable signal processing components.
3. Extract: the filterbank formula, the attention weighting over frequency or time,
   or the multi-scale decomposition computation.

## High-Value Query Angles for Audio

- Learnable filterbanks: `"learnable filterbank sinc convolution raw waveform audio"`
- Frequency-aware attention: `"frequency-aware attention audio spectrogram transformer"`
- Multi-scale audio: `"multi-scale temporal convolution audio representation learning"`
- Self-supervised audio: `"self-supervised audio representation contrastive learning architecture"`
- Fourier audio: `"Fourier transform audio neural network frequency domain learning"`
- Audio transformers: `"audio spectrogram transformer attention classification mechanism"`
- Waveform models: `"raw waveform convolutional architecture audio classification"`
- Temporal dilated: `"dilated convolution temporal audio speech recognition architecture"`

## Paper Selection Criteria

**Select if abstract contains:**
- Learnable filterbank or frequency decomposition
- Explicit frequency-time attention formulation
- Multi-scale or multi-resolution audio processing
- Novel waveform-domain operations

**Skip if abstract primarily contains:**
- Speech recognition benchmarks (LibriSpeech, CommonVoice) without architectural description
- Noise robustness papers without architectural innovation
- Music generation papers (different problem from classification/representation)

## Mechanism Extraction Focus

- The filterbank computation: how frequency bands are learned from waveform
- The spectrogram attention: how the model attends over frequency bins vs time frames
- Multi-scale decomposition: how different temporal resolutions are computed and fused
- Any causal constraint in the temporal processing (important for streaming)
