---
name: researcher-timeseries
description: arXiv query strategy and paper selection guidance for timeseries domain research. Load when running research for timeseries/forecasting tasks.
agent: researcher
version: 1.0
domains: [timeseries]
---

# Researcher Skill — Timeseries Domain

## Purpose

Timeseries research spans forecasting, anomaly detection, and classification. The most
useful papers for this pipeline are those that propose novel temporal modelling mechanisms:
decomposition methods, attention over time, dilated convolution structures, and
multi-scale temporal representations. Papers focused only on specific datasets
(electricity, traffic, weather) without architectural innovation have little to offer.

## Key Principles

- Query for architectural mechanisms, not forecasting benchmarks.
- Papers that decompose timeseries (trend/seasonality) into separate components produce
  the most extractable mechanisms.
- Attention-over-time papers often contain explicit temporal attention score formulas.

## Procedure

1. For each slot, use one query targeting temporal attention or decomposition mechanisms
   and one targeting convolutional or state-space approaches.
2. Prefer papers describing causal temporal operations — these are directly implementable.
3. Extract the decomposition formula, attention weighting over timesteps, or dilation schedule.

## High-Value Query Angles for Timeseries

- Temporal decomposition: `"trend seasonal decomposition neural network time series forecasting"`
- Dilated convolutions: `"dilated causal convolution temporal convolutional network architecture"`
- Temporal attention: `"temporal self-attention time series transformer forecasting mechanism"`
- Multi-scale temporal: `"multi-scale temporal representation time series hierarchical"`
- State space models: `"structured state space model time series sequence S4"`
- Normalisation for non-stationarity: `"instance normalization reversible non-stationary time series"`
- Patch-based Transformers: `"PatchTST temporal patch transformer forecasting"`
- Hybrid models: `"CNN LSTM hybrid temporal architecture forecasting"`

## Paper Selection Criteria

**Select if abstract contains:**
- Decomposition into components (trend, seasonal, residual)
- Causal attention formulation
- Multi-resolution or multi-scale representation
- "We propose a novel temporal/forecasting architecture/layer"

**Skip if abstract primarily contains:**
- Dataset-specific results only (ETT, Electricity, Traffic benchmarks) without mechanism
- "We apply a pre-trained model to timeseries"
- Survey papers

## Mechanism Extraction Focus

- The decomposition formula (e.g. how trend is extracted from the raw series)
- The causal attention scoring function (queries from current step, keys from history)
- The dilation schedule for causal convolutions
- Any normalisation applied per-sequence (instance norm computation)
