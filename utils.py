from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def build_distance_matrix(cities: np.ndarray) -> np.ndarray:
    n = cities.shape[0]
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cities[i, 0] - cities[j, 0]
            dy = cities[i, 1] - cities[j, 1]
            d = (dx ** 2 + dy ** 2) ** 0.5
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def compute_route_length(route: List[int], distance_matrix: np.ndarray) -> float:
    length = 0.0
    for i in range(len(route) - 1):
        length += distance_matrix[route[i], route[i + 1]]
    length += distance_matrix[route[-1], route[0]]
    return float(length)


def parse_route_input(route_str: str, labels: List[str]) -> Optional[List[int]]:
    if not route_str:
        return None

    # Replace common separators with a single comma
    normalized = route_str.replace("-", ",").replace(";", ",")
    parts = [p.strip().upper() for p in normalized.split(",") if p.strip()]

    if not parts:
        return None

    label_to_index = {label.upper(): idx for idx, label in enumerate(labels)}
    route: List[int] = []

    for token in parts:
        if token not in label_to_index:
            return None
        route.append(label_to_index[token])

    return route


def validate_route(route: List[int], num_cities: int) -> Tuple[bool, str]:
    if len(route) != num_cities:
        return (
            False,
            f"Your route must contain exactly {num_cities} cities. "
            f"Currently, it has {len(route)} entries.",
        )

    if min(route) < 0 or max(route) >= num_cities:
        return False, "Invalid city index detected."

    if len(set(route)) != num_cities:
        return False, "Each city must appear exactly once in your route."

    return True, "Route is valid."
