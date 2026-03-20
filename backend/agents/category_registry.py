"""
Category registry for Research Labs.

Maps the 8 supported modality categories to:
- Domain handlers
- Default base architectures
- Dataset types with metadata
- Available data sources (synthetic generators + public APIs)
"""
from __future__ import annotations

# ── Category → domain handler mapping ────────────────────────────────────────

CATEGORY_TO_DOMAINS: dict[str, list[str]] = {
    "text"                  : ["language", "generative"],
    "vision"                : ["vision", "generative"],
    "audio"                 : ["audio"],
    "timeseries"            : ["timeseries"],
    "graph"                 : ["graph"],
    "multimodal_text_image" : ["multimodal", "vision", "language"],
    "tabular"               : ["tabular"],
    "recommendation"        : ["recommendation", "graph"],
    "reinforcement_learning": ["rl"],
}

CATEGORY_TO_DEFAULT_ARCHITECTURES: dict[str, list[str]] = {
    "text"                  : ["transformer", "lstm"],
    "vision"                : ["cnn", "vit"],
    "audio"                 : ["conformer", "cnn_audio"],
    "timeseries"            : ["lstm_ts", "transformer_ts"],
    "graph"                 : ["gcn", "gat"],
    "multimodal_text_image" : ["clip", "flamingo"],
    "tabular"               : ["mlp", "resnet_tabular"],
    "recommendation"        : ["embedding_cf", "attention_rec"],
    "reinforcement_learning": ["dqn", "ppo"],
}

# ── Full category definitions ─────────────────────────────────────────────────

DATASET_CATEGORIES: dict[str, dict] = {
    "text": {
        "id"         : "text",
        "name"       : "Natural Language Processing",
        "description": "Text classification, generation, translation, QA, summarization, NER",
        "dataset_types": [
            {
                "type"       : "text_classification",
                "description": "Classify text into categories (sentiment, topic, intent)",
                "example_task": "Binary sentiment analysis on product reviews",
                "domains"    : ["language"],
                "recommended_architectures": ["transformer", "lstm", "cnn_text"],
            },
            {
                "type"       : "text_generation",
                "description": "Autoregressive language modeling and text generation",
                "example_task": "GPT-style next-token prediction on a text corpus",
                "domains"    : ["language", "generative"],
                "recommended_architectures": ["transformer", "lstm"],
            },
            {
                "type"       : "machine_translation",
                "description": "Sequence-to-sequence translation between languages",
                "example_task": "English → Spanish translation",
                "domains"    : ["language"],
                "recommended_architectures": ["transformer"],
            },
            {
                "type"       : "question_answering",
                "description": "Extractive or abstractive reading comprehension",
                "example_task": "SQuAD-style span extraction",
                "domains"    : ["language"],
                "recommended_architectures": ["transformer"],
            },
            {
                "type"       : "named_entity_recognition",
                "description": "Token-level sequence labeling for entities",
                "example_task": "Tag persons, organizations, locations in text",
                "domains"    : ["language"],
                "recommended_architectures": ["transformer", "lstm"],
            },
            {
                "type"       : "summarization",
                "description": "Abstractive or extractive document summarization",
                "example_task": "Summarize news articles to 1-2 sentences",
                "domains"    : ["language", "generative"],
                "recommended_architectures": ["transformer"],
            },
        ],
        "synthetic_generators": [
            {"id": "faker_text",    "name": "Faker Text",        "description": "Fake reviews and articles"},
            {"id": "template_based","name": "Template Generator", "description": "Template + variable substitution"},
            {"id": "random_tokens", "name": "Random Tokens",     "description": "Random token ID sequences"},
        ],
        "public_apis": [
            {"id": "huggingface_wikitext", "name": "WikiText-103",  "provider": "huggingface"},
            {"id": "huggingface_ag_news",  "name": "AG News",       "provider": "huggingface"},
            {"id": "huggingface_imdb",     "name": "IMDB Reviews",  "provider": "huggingface"},
        ],
    },

    "vision": {
        "id"         : "vision",
        "name"       : "Computer Vision",
        "description": "Image classification, object detection, segmentation, image generation",
        "dataset_types": [
            {
                "type"       : "image_classification",
                "description": "Assign a single label to an input image",
                "example_task": "Classify CIFAR-10 images into 10 categories",
                "domains"    : ["vision"],
                "recommended_architectures": ["cnn", "vit", "efficientnet"],
            },
            {
                "type"       : "object_detection",
                "description": "Locate and classify multiple objects in an image",
                "example_task": "Detect and box pedestrians in street scenes",
                "domains"    : ["vision"],
                "recommended_architectures": ["cnn", "detr"],
            },
            {
                "type"       : "semantic_segmentation",
                "description": "Pixel-level classification of every pixel in an image",
                "example_task": "Segment road, sky, buildings in driving scenes",
                "domains"    : ["vision"],
                "recommended_architectures": ["unet", "vit"],
            },
            {
                "type"       : "image_generation",
                "description": "Generate realistic images from noise or conditioning",
                "example_task": "Generate faces with a GAN or VAE",
                "domains"    : ["vision", "generative"],
                "recommended_architectures": ["gan", "vae", "diffusion"],
            },
        ],
        "synthetic_generators": [
            {"id": "random_noise",        "name": "Random Noise",      "description": "Uniform random pixel tensors"},
            {"id": "geometric_shapes",    "name": "Geometric Shapes",  "description": "Procedurally generated shapes"},
            {"id": "procedural_textures", "name": "Procedural Textures","description": "Noise-based texture generation"},
        ],
        "public_apis": [
            {"id": "huggingface_cifar10",    "name": "CIFAR-10",    "provider": "huggingface"},
            {"id": "huggingface_mnist",      "name": "MNIST",       "provider": "huggingface"},
            {"id": "huggingface_imagenet1k", "name": "ImageNet-1K", "provider": "huggingface"},
        ],
    },

    "audio": {
        "id"         : "audio",
        "name"       : "Audio & Speech",
        "description": "Speech recognition, speaker ID, music classification, audio tagging",
        "dataset_types": [
            {
                "type"       : "speech_recognition",
                "description": "Transcribe spoken audio to text",
                "example_task": "ASR on LibriSpeech clips",
                "domains"    : ["audio"],
                "recommended_architectures": ["conformer", "wav2vec"],
            },
            {
                "type"       : "speaker_identification",
                "description": "Identify or verify a speaker from audio",
                "example_task": "Identify 1000 speakers from 3-second clips",
                "domains"    : ["audio"],
                "recommended_architectures": ["conformer", "cnn_audio"],
            },
            {
                "type"       : "music_classification",
                "description": "Classify music by genre, mood, or instrument",
                "example_task": "Genre classification on GTZAN dataset",
                "domains"    : ["audio"],
                "recommended_architectures": ["cnn_audio", "transformer"],
            },
            {
                "type"       : "audio_tagging",
                "description": "Multi-label classification of environmental sounds",
                "example_task": "Tag AudioSet clips with sound event labels",
                "domains"    : ["audio"],
                "recommended_architectures": ["cnn_audio", "conformer"],
            },
        ],
        "synthetic_generators": [
            {"id": "sine_wave_patterns", "name": "Sine Waves",   "description": "Synthetic sine wave spectrograms"},
            {"id": "white_noise",        "name": "White Noise",  "description": "Random noise mel-spectrograms"},
        ],
        "public_apis": [
            {"id": "huggingface_librispeech", "name": "LibriSpeech", "provider": "huggingface"},
            {"id": "huggingface_gtzan",       "name": "GTZAN",       "provider": "huggingface"},
        ],
    },

    "timeseries": {
        "id"         : "timeseries",
        "name"       : "Time Series",
        "description": "Forecasting, anomaly detection, classification, imputation",
        "dataset_types": [
            {
                "type"       : "forecasting",
                "description": "Predict future values from historical sequences",
                "example_task": "Multi-step electricity demand forecasting",
                "domains"    : ["timeseries"],
                "recommended_architectures": ["lstm_ts", "transformer_ts", "tcn"],
            },
            {
                "type"       : "anomaly_detection",
                "description": "Detect outliers and anomalies in sequential data",
                "example_task": "Detect sensor failures in industrial IoT streams",
                "domains"    : ["timeseries"],
                "recommended_architectures": ["lstm_ts", "autoencoder_ts"],
            },
            {
                "type"       : "classification",
                "description": "Classify time series sequences into categories",
                "example_task": "Classify ECG signals as normal/abnormal",
                "domains"    : ["timeseries"],
                "recommended_architectures": ["lstm_ts", "transformer_ts"],
            },
        ],
        "synthetic_generators": [
            {"id": "random_walk",      "name": "Random Walk",     "description": "Brownian motion sequences"},
            {"id": "seasonal_patterns","name": "Seasonal Patterns","description": "Sinusoidal + trend + noise"},
        ],
        "public_apis": [
            {"id": "huggingface_etth",        "name": "ETTh1",      "provider": "huggingface"},
            {"id": "huggingface_electricity", "name": "Electricity", "provider": "huggingface"},
        ],
    },

    "graph": {
        "id"         : "graph",
        "name"       : "Graph Neural Networks",
        "description": "Node classification, link prediction, graph classification, community detection",
        "dataset_types": [
            {
                "type"       : "node_classification",
                "description": "Classify each node in a graph using its features and topology",
                "example_task": "Classify citation network nodes by paper topic",
                "domains"    : ["graph"],
                "recommended_architectures": ["gcn", "gat", "graphsage"],
            },
            {
                "type"       : "link_prediction",
                "description": "Predict whether an edge exists between two nodes",
                "example_task": "Recommend friends in a social network",
                "domains"    : ["graph"],
                "recommended_architectures": ["gcn", "gat"],
            },
            {
                "type"       : "graph_classification",
                "description": "Classify entire graphs (not just nodes)",
                "example_task": "Classify molecular graphs as toxic/non-toxic",
                "domains"    : ["graph"],
                "recommended_architectures": ["gin", "gcn"],
            },
        ],
        "synthetic_generators": [
            {"id": "barabasi_albert", "name": "Barabási–Albert", "description": "Scale-free random graphs"},
            {"id": "erdos_renyi",     "name": "Erdős–Rényi",     "description": "Random graphs with fixed edge probability"},
        ],
        "public_apis": [
            {"id": "pyg_cora",    "name": "Cora",    "provider": "torch_geometric"},
            {"id": "pyg_citeseer","name": "CiteSeer","provider": "torch_geometric"},
        ],
    },

    "multimodal_text_image": {
        "id"         : "multimodal_text_image",
        "name"       : "Multimodal (Text + Image)",
        "description": "Image captioning, visual QA, image-text retrieval, contrastive learning",
        "dataset_types": [
            {
                "type"       : "image_captioning",
                "description": "Generate natural language captions for images",
                "example_task": "Caption COCO images with a transformer decoder",
                "domains"    : ["multimodal", "vision", "language"],
                "recommended_architectures": ["clip", "flamingo"],
            },
            {
                "type"       : "visual_qa",
                "description": "Answer questions about images",
                "example_task": "Answer 'What color is the car?' given an image",
                "domains"    : ["multimodal"],
                "recommended_architectures": ["clip", "flamingo"],
            },
            {
                "type"       : "image_text_retrieval",
                "description": "Retrieve matching images for text queries and vice versa",
                "example_task": "CLIP-style contrastive image-text matching",
                "domains"    : ["multimodal"],
                "recommended_architectures": ["clip"],
            },
        ],
        "synthetic_generators": [
            {"id": "synthetic_pairs", "name": "Synthetic Pairs", "description": "Random image + token sequence pairs"},
        ],
        "public_apis": [
            {"id": "huggingface_coco_captions", "name": "COCO Captions", "provider": "huggingface"},
            {"id": "huggingface_flickr30k",     "name": "Flickr30k",     "provider": "huggingface"},
        ],
    },

    "tabular": {
        "id"         : "tabular",
        "name"       : "Tabular Data",
        "description": "Classification, regression, clustering, anomaly detection on structured data",
        "dataset_types": [
            {
                "type"       : "classification",
                "description": "Predict a discrete label from tabular features",
                "example_task": "Binary customer churn prediction",
                "domains"    : ["tabular"],
                "recommended_architectures": ["mlp", "resnet_tabular", "tabnet"],
            },
            {
                "type"       : "regression",
                "description": "Predict a continuous value from tabular features",
                "example_task": "House price prediction from property features",
                "domains"    : ["tabular"],
                "recommended_architectures": ["mlp", "resnet_tabular"],
            },
            {
                "type"       : "anomaly_detection",
                "description": "Detect unusual rows in structured datasets",
                "example_task": "Fraud detection in transaction data",
                "domains"    : ["tabular"],
                "recommended_architectures": ["autoencoder_tabular", "mlp"],
            },
        ],
        "synthetic_generators": [
            {"id": "sklearn_classification", "name": "sklearn Classification", "description": "make_classification synthetic data"},
            {"id": "sklearn_regression",     "name": "sklearn Regression",     "description": "make_regression synthetic data"},
            {"id": "faker_synthetic",        "name": "Faker Tabular",          "description": "Realistic fake tabular data via Faker"},
        ],
        "public_apis": [
            {"id": "huggingface_adult",    "name": "Adult Income",       "provider": "huggingface"},
            {"id": "huggingface_california","name": "California Housing", "provider": "huggingface"},
        ],
    },

    "reinforcement_learning": {
        "id"         : "reinforcement_learning",
        "name"       : "Reinforcement Learning",
        "description": "Policy optimization, value-based methods, actor-critic, model-based RL",
        "dataset_types": [
            {
                "type"       : "policy_optimization",
                "description": "Discover novel policy network architectures for continuous/discrete control",
                "example_task": "Find efficient actor-critic architectures for CartPole and LunarLander",
                "domains"    : ["rl"],
                "recommended_architectures": ["ppo", "a3c", "sac"],
            },
            {
                "type"       : "value_estimation",
                "description": "Discover value function architectures for Q-learning and TD methods",
                "example_task": "Find compact Q-network architectures for Atari environments",
                "domains"    : ["rl"],
                "recommended_architectures": ["dqn", "dueling_dqn", "rainbow"],
            },
            {
                "type"       : "model_based",
                "description": "Discover world model architectures that learn environment dynamics",
                "example_task": "Find sample-efficient world models for MuJoCo locomotion tasks",
                "domains"    : ["rl"],
                "recommended_architectures": ["dreamer", "mbpo"],
            },
        ],
        "synthetic_generators": [
            {"id": "gym_cartpole",   "name": "CartPole-v1",    "description": "Classic pole balancing control task"},
            {"id": "gym_lunarlander","name": "LunarLander-v2", "description": "Continuous lunar landing environment"},
            {"id": "gym_mountaincar","name": "MountainCar",    "description": "Sparse reward hill-climbing task"},
        ],
        "public_apis": [
            {"id": "gym_atari",   "name": "Atari (ALE)",   "provider": "gymnasium"},
            {"id": "gym_mujoco",  "name": "MuJoCo",        "provider": "gymnasium"},
        ],
    },

    "recommendation": {
        "id"         : "recommendation",
        "name"       : "Recommendation Systems",
        "description": "Collaborative filtering, content-based, knowledge graph recommendations",
        "dataset_types": [
            {
                "type"       : "collaborative_filtering",
                "description": "Learn user-item preferences from interaction history",
                "example_task": "Movie recommendations from ratings matrix",
                "domains"    : ["recommendation"],
                "recommended_architectures": ["embedding_cf", "ncf"],
            },
            {
                "type"       : "content_based",
                "description": "Recommend items based on content features and user profiles",
                "example_task": "News article recommendations based on text similarity",
                "domains"    : ["recommendation", "language"],
                "recommended_architectures": ["attention_rec", "embedding_cf"],
            },
        ],
        "synthetic_generators": [
            {"id": "random_ratings",  "name": "Random Ratings",  "description": "Sparse random user-item ratings matrix"},
            {"id": "biased_ratings",  "name": "Biased Ratings",  "description": "Ratings with user/item popularity bias"},
        ],
        "public_apis": [
            {"id": "huggingface_movielens", "name": "MovieLens-1M", "provider": "huggingface"},
        ],
    },
}


# ── Helper functions ──────────────────────────────────────────────────────────

def get_category(category_id: str) -> dict | None:
    return DATASET_CATEGORIES.get(category_id)


def get_all_categories() -> list[dict]:
    return list(DATASET_CATEGORIES.values())


def get_domains_for_category(category_id: str) -> list[str]:
    return CATEGORY_TO_DOMAINS.get(category_id, [])


def get_default_architectures(category_id: str) -> list[str]:
    return CATEGORY_TO_DEFAULT_ARCHITECTURES.get(category_id, [])


def infer_domains(category_id: str, preferred_domains: list[str] | None = None) -> list[str]:
    """Return domains to use for this session, respecting user preference if set."""
    all_domains = get_domains_for_category(category_id)
    if preferred_domains:
        return [d for d in preferred_domains if d in all_domains] or all_domains
    return all_domains
