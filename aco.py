from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class AntColony:

    distance_matrix: np.ndarray
    num_ants: int = 20
    num_iterations: int = 100
    alpha: float = 1.0
    beta: float = 5.0
    evaporation_rate: float = 0.5
    q: float = 100.0
    start_city: int = 0

    def __post_init__(self) -> None:
        self.num_cities = self.distance_matrix.shape[0]
        # Initialize pheromone: small positive constant on each edge
        self.pheromone = np.ones_like(self.distance_matrix, dtype=float)
        # 1 / distance is our heuristic (with zero on diagonal to avoid division by zero)
        with np.errstate(divide="ignore"):
            self.heuristic = 1.0 / self.distance_matrix
        self.heuristic[self.distance_matrix == 0] = 0.0

        # Track best tour found
        self.best_route: List[int] = []
        self.best_distance: float = float("inf")
        self.best_history: List[float] = []

    def run(self) -> Tuple[List[int], float, List[float]]:
        for _ in range(self.num_iterations):
            all_routes = []
            all_lengths = []

            # Each ant constructs a route
            for _ant in range(self.num_ants):
                route = self._build_route()
                length = self._route_length(route)
                all_routes.append(route)
                all_lengths.append(length)

                # Keep track of global best
                if length < self.best_distance:
                    self.best_distance = length
                    self.best_route = route

            # Update pheromones based on ants' routes
            self._update_pheromones(all_routes, all_lengths)

            # Store progress
            self.best_history.append(self.best_distance)

        return self.best_route, self.best_distance, self.best_history

    def _build_route(self) -> List[int]:
        route = [self.start_city]
        unvisited = set(range(self.num_cities)) - {self.start_city}

        current_city = self.start_city
        while unvisited:
            next_city = self._choose_next_city(current_city, unvisited)
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        return route

    def _choose_next_city(self, current_city: int, unvisited: set[int]) -> int:
        """Choose the next city based on pheromone and heuristic information."""
        unvisited_list = list(unvisited)
        pheromone = self.pheromone[current_city, unvisited_list]
        heuristic = self.heuristic[current_city, unvisited_list]

        # Compute desirability: tau^alpha * eta^beta
        desirability = (pheromone ** self.alpha) * (heuristic ** self.beta)

        # If due to numeric issues desirability sums to zero, choose randomly
        if np.all(desirability == 0):
            return np.random.choice(unvisited_list)

        probabilities = desirability / desirability.sum()
        # Choose according to the probability distribution
        next_city = np.random.choice(unvisited_list, p=probabilities)
        return int(next_city)

    def _route_length(self, route: List[int]) -> float:
        """Compute the length of a tour, including return to the starting city."""
        length = 0.0
        for i in range(len(route) - 1):
            length += self.distance_matrix[route[i], route[i + 1]]
        # Return to start
        length += self.distance_matrix[route[-1], route[0]]
        return float(length)

    def _update_pheromones(self, routes: List[List[int]], lengths: List[float]) -> None:
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)

        # Deposition from each ant
        for route, length in zip(routes, lengths):
            deposit_amount = self.q / length
            for i in range(len(route) - 1):
                a = route[i]
                b = route[i + 1]
                self.pheromone[a, b] += deposit_amount
                self.pheromone[b, a] += deposit_amount  # symmetric TSP

            # Also deposit on the return edge to the start
            a = route[-1]
            b = route[0]
            self.pheromone[a, b] += deposit_amount
            self.pheromone[b, a] += deposit_amount
