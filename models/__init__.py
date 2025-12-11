"""Model wrappers for VLM experiments."""

from typing import Dict, List, Type

from .model import AbstractModel
from .qwen3inst import Qwen3Instruct
from .qwen3think import Qwen3Think
from .qwen25inst import Qwen25Instruct
from .qwen25_32b import Qwen25Instruct32B
from .llava import Llava


MODEL_REGISTRY: Dict[str, Type[AbstractModel]] = {
    "llava": Llava,
    "qwen3inst": Qwen3Instruct,
    "qwen3think": Qwen3Think,
    "qwen25inst": Qwen25Instruct,
    "qwen25_32b": Qwen25Instruct32B,
}


def valid_model_names() -> List[str]:
    """Return the list of model identifiers supported by the registry."""
    return sorted(MODEL_REGISTRY.keys())


def load_model_by_name(model_name: str, **load_kwargs) -> AbstractModel:
    """
    Load a model wrapper by its registry key.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g., 'qwen3inst').
        **load_kwargs: Keyword arguments forwarded to the wrapper's load_model.

    Returns:
        Instantiated model wrapper.

    Raises:
        ValueError: If model_name is not registered.
    """
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        valid = ", ".join(valid_model_names())
        raise ValueError(
            f"Unknown model_name '{model_name}'. Valid options: {valid}"
        ) from exc
    return model_cls.load_model(**load_kwargs)


__all__ = [
    "AbstractModel",
    "Qwen3Instruct",
    "Qwen3Think",
    "Qwen25Instruct",
    "Qwen25Instruct32B",
    "Llava",
    "MODEL_REGISTRY",
    "load_model_by_name",
    "valid_model_names",
]
