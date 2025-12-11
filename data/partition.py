"""Utilities for slicing dataset splits into evenly sized partitions."""

from typing import List, Sequence, Tuple, TypeVar

InputT = TypeVar("InputT")
LabelT = TypeVar("LabelT")

PartitionedSplit = Tuple[List[InputT], List[LabelT]]

_VALID_PARTITIONS = {"all", "a", "b"}


def _normalize_partition(partition: str) -> str:
    """Normalize user-provided partition name and validate it."""
    if partition is None:
        return "all"
    normalized = partition.strip().lower()
    if normalized not in _VALID_PARTITIONS:
        valid = "', '".join(sorted(_VALID_PARTITIONS))
        raise ValueError(f"Invalid partition '{partition}'. Expected one of '{valid}'.")
    return normalized


def apply_partition(
    inputs: Sequence[InputT],
    labels: Sequence[LabelT],
    partition: str,
    split_name: str = "split",
) -> PartitionedSplit:
    """
    Slice inputs/labels into A/B partitions while preserving ordering.

    Args:
        inputs: Sequence of samples/prompts.
        labels: Sequence of labels/targets matching inputs.
        partition: 'all', 'a', or 'b' (case-insensitive).
        split_name: Friendly name for error messages.

    Returns:
        Tuple (partition_inputs, partition_labels) as lists.
    """
    normalized = _normalize_partition(partition)
    if normalized == "all":
        return list(inputs), list(labels)

    total = len(inputs)
    midpoint = (total + 1) // 2  # ensures near-even halves, A gets extra item when odd

    if normalized == "a":
        subset_inputs = inputs[:midpoint]
        subset_labels = labels[:midpoint]
    else:
        subset_inputs = inputs[midpoint:]
        subset_labels = labels[midpoint:]

    if len(subset_inputs) == 0:
        raise ValueError(
            f"{split_name} partition '{partition}' is empty. "
            "Dataset split may be too small for partitioning."
        )

    return list(subset_inputs), list(subset_labels)

