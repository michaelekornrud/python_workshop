"""Exercises for Pythonic thinking: collections, comprehensions, unpacking, truthiness, EAFP.

Implement the functions below to make tests pass. Prefer readable, idiomatic solutions.
"""
from __future__ import annotations
from typing import Any, Iterable, Iterator, List, Sequence, Tuple


def normalize_whitespace(s: str) -> str:
    """Return a string where runs of whitespace are collapsed to single spaces.

    Example: "  a\t b\n c  " -> "a b c"
    """
    raise NotImplementedError


def unique_preserve_order(items: Iterable[Any]) -> List[Any]:
    """Return unique items in first-seen order.

    Hint: a set for membership + a list for order.
    """
    raise NotImplementedError


def pairwise_sum(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """Return element-wise sums for pairs from a and b, truncating to the shorter.

    Prefer zip over indexing.
    """
    raise NotImplementedError


def transpose(matrix: Sequence[Sequence[Any]]) -> List[List[Any]]:
    """Transpose a rectangular 2D matrix.

    Use zip(*) and consider converting tuples to lists.
    """
    raise NotImplementedError


def head_tail(seq: Sequence[Any]) -> Tuple[Any, List[Any]]:
    """Return the first element and the remaining elements as a list.

    Use sequence unpacking.
    """
    raise NotImplementedError


def safe_get(mapping: dict, path: Sequence[Any], default: Any | None = None) -> Any | None:
    """EAFP-style nested dict lookup.

    Given a path like ["a", "b", 0], navigate mapping["a"]["b"][0]. If any step fails,
    return default. Do not use if-chains; use try/except.
    """
    raise NotImplementedError


def flatten_once(nested: Iterable[Iterable[Any]]) -> List[Any]:
    """Flatten by one level: [[1,2],[3]] -> [1,2,3]

    Use a comprehension.
    """
    raise NotImplementedError