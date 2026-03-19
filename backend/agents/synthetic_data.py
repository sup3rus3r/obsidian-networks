"""
Synthetic data generators for all 8 supported modalities.

Each generator returns (X_train, X_test, y_train, y_test) tuples
(or domain-equivalent structures) using only installed packages.

No real datasets are downloaded here — all data is procedurally generated.
"""
from __future__ import annotations

import numpy as np
import random as _random


# ── Tabular ───────────────────────────────────────────────────────────────────

def generate_tabular(
    size: int = 1000,
    n_features: int = 20,
    task: str = "classification",   # "classification" | "regression"
    n_classes: int = 2,
    seed: int = 42,
) -> tuple:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(seed)
    if task == "regression":
        X, y = make_regression(n_samples=size, n_features=n_features, noise=0.1, random_state=seed)
        y = y.astype(np.float32)
    else:
        X, y = make_classification(
            n_samples=size, n_features=n_features,
            n_classes=n_classes, n_informative=max(2, n_features // 2),
            random_state=seed,
        )
        y = y.astype(np.int64)

    X = X.astype(np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=seed)


# ── Vision (images) ───────────────────────────────────────────────────────────

def generate_image(
    size: int = 1000,
    resolution: int = 32,
    n_classes: int = 10,
    channels: int = 3,
    seed: int = 42,
) -> tuple:
    rng = np.random.RandomState(seed)
    split = int(size * 0.8)
    X = rng.uniform(0, 1, (size, resolution, resolution, channels)).astype(np.float32)
    y = rng.randint(0, n_classes, size).astype(np.int64)
    return X[:split], X[split:], y[:split], y[split:]


# ── Audio ─────────────────────────────────────────────────────────────────────

def generate_audio(
    size: int = 1000,
    sample_rate: int = 16000,
    duration_s: float = 1.0,
    n_classes: int = 10,
    seed: int = 42,
) -> tuple:
    """Generate synthetic audio as mel-spectrograms (n_mels, time_frames)."""
    rng   = np.random.RandomState(seed)
    n_mels = 64
    frames = int(duration_s * sample_rate / 512)  # hop_length=512
    split  = int(size * 0.8)

    # Sine wave + noise spectrograms
    X = rng.uniform(0, 1, (size, n_mels, frames)).astype(np.float32)
    y = rng.randint(0, n_classes, size).astype(np.int64)
    return X[:split], X[split:], y[:split], y[split:]


# ── Time Series ───────────────────────────────────────────────────────────────

def generate_timeseries(
    size: int = 1000,
    seq_len: int = 50,
    n_features: int = 1,
    task: str = "forecasting",     # "forecasting" | "classification"
    n_classes: int = 5,
    forecast_horizon: int = 10,
    seed: int = 42,
) -> tuple:
    rng   = np.random.RandomState(seed)
    split = int(size * 0.8)

    # Random walk time series
    noise = rng.randn(size, seq_len + forecast_horizon, n_features).astype(np.float32)
    X_all = np.cumsum(noise, axis=1)

    if task == "forecasting":
        X = X_all[:, :seq_len, :]
        y = X_all[:, seq_len:, 0]            # predict next `forecast_horizon` steps
    else:
        X = X_all[:, :seq_len, :]
        y = rng.randint(0, n_classes, size).astype(np.int64)

    return X[:split], X[split:], y[:split], y[split:]


# ── Graph ─────────────────────────────────────────────────────────────────────

def generate_graph(
    n_nodes: int = 200,
    n_classes: int = 5,
    n_features: int = 16,
    task: str = "node_classification",
    seed: int = 42,
) -> dict:
    """
    Returns a dict with node_features, edge_index, labels.
    Compatible with torch_geometric Data format.
    """
    import networkx as nx

    rng = np.random.RandomState(seed)
    G   = nx.barabasi_albert_graph(n_nodes, m=3, seed=seed)

    node_features = rng.randn(n_nodes, n_features).astype(np.float32)
    labels        = rng.randint(0, n_classes, n_nodes).astype(np.int64)

    edges     = list(G.edges())
    edge_index = np.array(edges, dtype=np.int64).T   # shape (2, E)

    # Train/val/test masks
    indices = np.random.permutation(n_nodes)
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask   = np.zeros(n_nodes, dtype=bool)
    test_mask  = np.zeros(n_nodes, dtype=bool)
    train_mask[indices[:int(n_nodes * 0.6)]] = True
    val_mask[indices[int(n_nodes * 0.6):int(n_nodes * 0.8)]] = True
    test_mask[indices[int(n_nodes * 0.8):]] = True

    return {
        "node_features": node_features,
        "edge_index"   : edge_index,
        "labels"       : labels,
        "train_mask"   : train_mask,
        "val_mask"     : val_mask,
        "test_mask"    : test_mask,
        "n_nodes"      : n_nodes,
        "n_features"   : n_features,
        "n_classes"    : n_classes,
    }


# ── Text / NLP ────────────────────────────────────────────────────────────────

def generate_text(
    size: int = 1000,
    seq_len: int = 128,
    vocab_size: int = 10000,
    n_classes: int = 5,
    task: str = "classification",  # "classification" | "generation"
    seed: int = 42,
) -> tuple:
    """Token ID sequences for transformer/LSTM models."""
    rng   = np.random.RandomState(seed)
    split = int(size * 0.8)

    X = rng.randint(1, vocab_size, (size, seq_len)).astype(np.int64)

    if task == "generation":
        # Next-token prediction: input is X[:, :-1], target is X[:, 1:]
        y = X[:, 1:].copy()
        X = X[:, :-1]
    else:
        y = rng.randint(0, n_classes, size).astype(np.int64)

    return X[:split], X[split:], y[:split], y[split:]


# ── Multimodal (image + text) ─────────────────────────────────────────────────

def generate_multimodal(
    size: int = 500,
    resolution: int = 32,
    seq_len: int = 64,
    vocab_size: int = 5000,
    embed_dim: int = 128,
    seed: int = 42,
) -> dict:
    """
    Returns paired image + text embeddings for CLIP-style contrastive learning.
    """
    rng   = np.random.RandomState(seed)
    split = int(size * 0.8)

    images = rng.randn(size, resolution, resolution, 3).astype(np.float32)
    tokens = rng.randint(1, vocab_size, (size, seq_len)).astype(np.int64)
    labels = np.arange(size, dtype=np.int64)   # each image-text pair is its own class

    return {
        "train": {
            "images": images[:split],
            "tokens": tokens[:split],
            "labels": labels[:split],
        },
        "test": {
            "images": images[split:],
            "tokens": tokens[split:],
            "labels": labels[split:],
        },
        "embed_dim"  : embed_dim,
        "vocab_size" : vocab_size,
        "resolution" : resolution,
    }


# ── Recommendation ────────────────────────────────────────────────────────────

def generate_recommendation(
    n_users: int = 500,
    n_items: int = 1000,
    density: float = 0.02,
    seed: int = 42,
) -> dict:
    """
    Sparse ratings matrix → (user_ids, item_ids, ratings) triplets.
    """
    rng    = np.random.RandomState(seed)
    n_obs  = int(n_users * n_items * density)
    user_ids = rng.randint(0, n_users, n_obs).astype(np.int64)
    item_ids = rng.randint(0, n_items, n_obs).astype(np.int64)
    ratings  = rng.uniform(1.0, 5.0, n_obs).astype(np.float32)

    split = int(n_obs * 0.8)
    return {
        "train": {
            "user_ids": user_ids[:split],
            "item_ids": item_ids[:split],
            "ratings" : ratings[:split],
        },
        "test": {
            "user_ids": user_ids[split:],
            "item_ids": item_ids[split:],
            "ratings" : ratings[split:],
        },
        "n_users": n_users,
        "n_items": n_items,
    }


# ── Generative (noise tensors) ────────────────────────────────────────────────

def generate_generative(
    size: int = 1000,
    resolution: int = 32,
    latent_dim: int = 128,
    channels: int = 3,
    seed: int = 42,
) -> dict:
    """Real images + latent noise vectors for GAN/VAE/Diffusion training."""
    rng   = np.random.RandomState(seed)
    split = int(size * 0.8)

    real_images = rng.uniform(0, 1, (size, resolution, resolution, channels)).astype(np.float32)
    noise       = rng.randn(size, latent_dim).astype(np.float32)

    return {
        "train": {"images": real_images[:split], "noise": noise[:split]},
        "test" : {"images": real_images[split:], "noise": noise[split:]},
        "latent_dim": latent_dim,
        "resolution": resolution,
        "channels"  : channels,
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

GENERATORS = {
    "tabular"       : generate_tabular,
    "vision"        : generate_image,
    "audio"         : generate_audio,
    "timeseries"    : generate_timeseries,
    "graph"         : generate_graph,
    "text"          : generate_text,
    "multimodal"    : generate_multimodal,
    "recommendation": generate_recommendation,
    "generative"    : generate_generative,
}


def get_synthetic_data(category: str, size: int = 1000, params: dict | None = None):
    """
    Generate synthetic data for the given category.

    Returns whatever the domain-specific generator produces
    (tuple for most, dict for graph/multimodal/recommendation/generative).
    """
    params = params or {}
    generator = GENERATORS.get(category)
    if not generator:
        raise ValueError(f"Unknown category: {category!r}. Valid: {list(GENERATORS)}")
    return generator(size=size, **{k: v for k, v in params.items() if k != "size"})
