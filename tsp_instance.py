from typing import List

import numpy as np


def generate_cities(num_cities: int = 10, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cities = rng.random((num_cities, 2))
    return cities


def get_city_labels(num_cities: int) -> List[str]:
    labels: List[str] = []
    alphabet = [chr(ord("A") + i) for i in range(26)]

    count = num_cities
    prefix_index = 0

    while count > 0:
        for letter in alphabet:
            if count <= 0:
                break
            if prefix_index == 0:
                labels.append(letter)
            else:
                labels.append(alphabet[prefix_index - 1] + letter)
            count -= 1
        prefix_index += 1

    return labels[:num_cities]
