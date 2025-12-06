import time
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import streamlit as st

from aco import AntColony
from tsp_instance import generate_cities, get_city_labels
from utils import (
    build_distance_matrix,
    compute_route_length,
)


def init_session_state() -> None:
    """Initialize all keys in Streamlit's session_state used by the app."""
    defaults = {
        "num_cities": 10,
        "seed": 42,
        "cities": None,
        "city_labels": None,
        "distance_matrix": None,
        # user_route = order of clicked cities (can be partial)
        "user_route": [],
        "user_distance": None,
        "aco_best_route": None,
        "aco_best_distance": None,
        "aco_best_history": None,
        "aco_params": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def regenerate_cities(num_cities: int, seed: int) -> None:
    """Generate a new TSP instance and store it in session_state."""
    cities = generate_cities(num_cities=num_cities, seed=seed)
    labels = get_city_labels(num_cities)
    dist_matrix = build_distance_matrix(cities)

    st.session_state.cities = cities
    st.session_state.city_labels = labels
    st.session_state.distance_matrix = dist_matrix
    st.session_state.user_route = []
    st.session_state.user_distance = None
    st.session_state.aco_best_route = None
    st.session_state.aco_best_distance = None
    st.session_state.aco_best_history = None


# ---------- helpers for drawing routes ----------


def closed_loop(route: List[int]) -> List[int]:
    """Return the route closed by returning to the start city."""
    if not route:
        return route
    if route[0] != route[-1]:
        return route + [route[0]]
    return route


def draw_route_with_arrows(
    ax: plt.Axes,
    cities: np.ndarray,
    route: List[int],
    color: str,
    max_segments: Optional[int] = None,
    arrow_every: int = 1,
) -> None:
    """
    Draw a connected polyline with arrowheads along the route.

    Args:
        ax: Matplotlib axes to draw on.
        cities: Array of city coordinates.
        route: List of city indices.
        color: Color for line and arrows.
        max_segments: If given, only draw the first `max_segments` segments
                      (used for animation).
        arrow_every: Draw an arrow on every Nth segment.
    """
    route_loop = closed_loop(route)
    if len(route_loop) < 2:
        return

    num_segments = len(route_loop) - 1
    if max_segments is None or max_segments > num_segments:
        max_segments = num_segments

    # Draw continuous line for the visible part of the route
    line_indices = route_loop[: max_segments + 1]
    xs = [cities[i, 0] for i in line_indices]
    ys = [cities[i, 1] for i in line_indices]
    ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9, zorder=2)

    # Draw arrowheads on segments
    for seg_idx in range(0, max_segments, arrow_every):
        start_idx = route_loop[seg_idx]
        end_idx = route_loop[seg_idx + 1]
        x_start, y_start = cities[start_idx]
        x_end, y_end = cities[end_idx]

        dx = x_end - x_start
        dy = y_end - y_start

        # place arrowhead a bit before the endpoint
        arrow_frac = 0.85
        x_arrow = x_start + dx * arrow_frac
        y_arrow = y_start + dy * arrow_frac

        ax.annotate(
            "",
            xy=(x_arrow, y_arrow),
            xytext=(x_start, y_start),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2,
                mutation_scale=12,
            ),
            zorder=3,
        )


def plot_routes(
    cities: np.ndarray,
    labels: List[str],
    user_route: Optional[List[int]] = None,
    aco_route: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Plot the cities and (optionally) the user and ACO routes.

    - Cities are shown as points with labels (A, B, C, ...).
    - Routes are drawn as continuous lines with arrowheads to show direction.
    """
    fig, ax = plt.subplots()

    x = cities[:, 0]
    y = cities[:, 1]
    ax.scatter(x, y, zorder=4)

    # Label each city
    for i, label in enumerate(labels):
        ax.text(x[i] + 0.01, y[i] + 0.01, label, fontsize=9, zorder=5)

    # Draw user route (blue)
    if user_route:
        draw_route_with_arrows(ax, cities, user_route, color="tab:blue")

    # Draw ACO best route (orange)
    if aco_route:
        draw_route_with_arrows(ax, cities, aco_route, color="tab:orange")

    ax.set_title("TSP Cities and Routes")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # Custom legend
    legend_handles = []
    if user_route:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="tab:blue",
                linewidth=2,
                marker=">",
                markersize=6,
                label="Your route",
            )
        )
    if aco_route:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="tab:orange",
                linewidth=2,
                marker=">",
                markersize=6,
                label="ACO best route",
            )
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best")

    fig.tight_layout()
    return fig


def animate_route(
    cities: np.ndarray,
    labels: List[str],
    route: List[int],
    color: str,
    title: str,
    delay: float = 0.4,
) -> None:
    """
    Animate a route being drawn step-by-step with arrowheads.

    Uses a Streamlit placeholder to update the plot in a loop.
    """
    if not route:
        return

    route_loop = closed_loop(route)
    num_segments = len(route_loop) - 1

    placeholder = st.empty()

    for k in range(1, num_segments + 1):
        fig, ax = plt.subplots()

        # Plot cities and labels
        x = cities[:, 0]
        y = cities[:, 1]
        ax.scatter(x, y, zorder=4)
        for i, label in enumerate(labels):
            ax.text(x[i] + 0.01, y[i] + 0.01, label, fontsize=9, zorder=5)

        # Draw only the first k segments
        draw_route_with_arrows(
            ax,
            cities,
            route,
            color=color,
            max_segments=k,
            arrow_every=1,
        )

        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title(title)
        fig.tight_layout()

        placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)
        time.sleep(delay)


def plot_aco_progress(best_history: List[float]) -> plt.Figure:
    """Plot the improvement of the best tour length over iterations."""
    fig, ax = plt.subplots()
    iterations = list(range(1, len(best_history) + 1))
    ax.plot(iterations, best_history, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best tour length so far")
    ax.set_title("Ant Colony Optimization Progress")
    fig.tight_layout()
    return fig


def show_intro() -> None:
    """Display introductory text explaining the game and TSP/ACO basics."""
    st.markdown(
        """
# Ant Colony TSP Learning Game

**Goal:** Visit all cities exactly once and return to the starting city,  
trying to make the total route as short as possible.

Then we run the **Ant Colony Optimization (ACO)** algorithm on the same map and compare:
- Your route vs. the algorithm’s best route.
- How the best route improves over iterations.

### How to play (click version)

1. Look at the map of cities (labeled A, B, C, ...).
2. Click the **city buttons** (A, B, C, ...) in the order you want to visit them.
   - A line with arrows will be drawn following your clicks.
3. After you clicked **all cities once**, your route is complete:
   - The app will compute your total route length.
4. Choose ACO parameters in the sidebar (or use defaults).
5. Click **Run Ant Colony** to see:
   - The best route the ants found.
   - A chart of best distance over iterations.
6. Use the **animation buttons** to watch routes being drawn step-by-step.
        """
    )


def handle_city_click(city_idx: int) -> None:
    """
    Update the user's route when a city button is clicked.

    - Adds the city to the route if not already visited.
    - When all cities have been chosen, computes route length.
    """
    labels = st.session_state.city_labels
    num_cities = len(labels)

    # start from existing route or empty
    route: List[int] = list(st.session_state.user_route or [])

    # ignore if already chosen
    if city_idx in route:
        return

    route.append(city_idx)
    st.session_state.user_route = route
    st.session_state.user_distance = None  # reset until route complete

    # When all cities chosen once, compute distance
    if len(route) == num_cities:
        dist = compute_route_length(route, st.session_state.distance_matrix)
        st.session_state.user_distance = dist
        st.success(
            f"Route complete! Length (including return to start): **{dist:.3f}**"
        )


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Ant Colony TSP Learning Game", layout="wide")
    init_session_state()

    # Sidebar controls
    st.sidebar.header("TSP Map Settings")
    num_cities = st.sidebar.slider("Number of cities", 8, 15, st.session_state.num_cities)
    seed = st.sidebar.number_input("Random seed", min_value=0, value=st.session_state.seed, step=1)

    if st.sidebar.button("Generate New Map"):
        st.session_state.num_cities = num_cities
        st.session_state.seed = seed
        regenerate_cities(num_cities, seed)

    # Make sure we have cities generated
    if st.session_state.cities is None:
        regenerate_cities(st.session_state.num_cities, st.session_state.seed)

    # ACO parameters
    st.sidebar.header("Ant Colony Parameters")
    num_ants = st.sidebar.slider("Number of ants", 5, 50, 20)
    num_iterations = st.sidebar.slider("Iterations", 10, 200, 60, step=10)
    alpha = st.sidebar.slider("Alpha (pheromone importance)", 0.1, 5.0, 1.0, step=0.1)
    beta = st.sidebar.slider("Beta (distance importance)", 0.1, 10.0, 5.0, step=0.1)
    evaporation_rate = st.sidebar.slider("Evaporation rate (rho)", 0.01, 0.99, 0.5, step=0.01)
    q = st.sidebar.slider("Pheromone deposit factor (Q)", 10.0, 500.0, 100.0, step=10.0)

    st.session_state.aco_params = {
        "num_ants": num_ants,
        "num_iterations": num_iterations,
        "alpha": alpha,
        "beta": beta,
        "evaporation_rate": evaporation_rate,
        "q": q,
    }

    show_intro()

    cities = st.session_state.cities
    labels = st.session_state.city_labels
    dist_matrix = st.session_state.distance_matrix

    col_left, col_right = st.columns([1.2, 1.0])

    # LEFT: Map
    with col_left:
        st.subheader("City Map & Routes")
        fig_map = plot_routes(
            cities=cities,
            labels=labels,
            user_route=st.session_state.user_route,
            aco_route=st.session_state.aco_best_route,
        )
        st.pyplot(fig_map)

    # RIGHT: Click interface for route
    with col_right:
        st.subheader("Build your route by clicking cities")

        st.markdown(
            "Click each city **once** in the order you want to visit them.\n\n"
            "When you have clicked all cities, the route will be closed into a circle automatically."
        )

        # Buttons for each city, arranged in columns
        num_buttons_cols = min(5, len(labels))
        cols = st.columns(num_buttons_cols)

        for idx, label in enumerate(labels):
            col = cols[idx % num_buttons_cols]
            with col:
                # disable button if city already chosen
                disabled = idx in (st.session_state.user_route or [])
                if st.button(
                    label,
                    key=f"city_btn_{idx}",
                    disabled=disabled,
                ):
                    handle_city_click(idx)

        # Reset button
        if st.button("Reset my route"):
            st.session_state.user_route = []
            st.session_state.user_distance = None

        # Show current order
        if st.session_state.user_route:
            route_labels = [labels[i] for i in st.session_state.user_route]
            st.write("Current order:", " → ".join(route_labels))
            remaining = len(labels) - len(st.session_state.user_route)
            if remaining > 0:
                st.info(f"Choose **{remaining}** more city/cities to complete the route.")
        else:
            st.info("Start by clicking a city button above.")

        # Show final distance if complete
        if st.session_state.user_distance is not None:
            st.success(
                f"Your route length (including return to start): "
                f"**{st.session_state.user_distance:.3f}**"
            )

        st.markdown("---")
        run_aco_clicked = st.button("Run Ant Colony")

        if run_aco_clicked:
            if not st.session_state.user_route or len(st.session_state.user_route) != len(labels):
                st.warning("Tip: you can still run ACO even if your route isn't complete, "
                           "but your comparison will only make sense once you've chosen all cities.")
            with st.spinner("Ants are exploring routes..."):
                aco = AntColony(
                    distance_matrix=dist_matrix,
                    num_ants=num_ants,
                    num_iterations=num_iterations,
                    alpha=alpha,
                    beta=beta,
                    evaporation_rate=evaporation_rate,
                    q=q,
                    start_city=
