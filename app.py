import string
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from aco import AntColony
from tsp_instance import generate_cities, get_city_labels
from utils import (
    build_distance_matrix,
    compute_route_length,
    parse_route_input,
    validate_route,
)


def init_session_state() -> None:
    """Initialize all keys in Streamlit's session_state used by the app."""
    defaults = {
        "num_cities": 10,
        "seed": 42,
        "cities": None,
        "city_labels": None,
        "distance_matrix": None,
        "user_route": None,
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
    st.session_state.user_route = None
    st.session_state.user_distance = None
    st.session_state.aco_best_route = None
    st.session_state.aco_best_distance = None
    st.session_state.aco_best_history = None


def plot_routes(
    cities: np.ndarray,
    labels: List[str],
    user_route: Optional[List[int]] = None,
    aco_route: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Plot the cities and (optionally) the user and ACO routes.

    - Cities are shown as points with labels (A, B, C, ...).
    - User route is drawn as one polyline.
    - ACO best route is drawn as another polyline.
    """
    fig, ax = plt.subplots()

    x = cities[:, 0]
    y = cities[:, 1]
    ax.scatter(x, y)

    # Label each city
    for i, label in enumerate(labels):
        ax.text(x[i] + 0.01, y[i] + 0.01, label, fontsize=9)

    # Helper to close the loop (return to start)
    def closed_loop(route: List[int]) -> List[int]:
        if not route:
            return route
        if route[0] != route[-1]:
            return route + [route[0]]
        return route

    # Plot user route
    if user_route:
        user_loop = closed_loop(user_route)
        ux = cities[user_loop, 0]
        uy = cities[user_loop, 1]
        ax.plot(ux, uy, linestyle="-", marker="o", label="Your route", alpha=0.7)

    # Plot ACO route
    if aco_route:
        aco_loop = closed_loop(aco_route)
        axx = cities[aco_loop, 0]
        ayy = cities[aco_loop, 1]
        ax.plot(
            axx,
            ayy,
            linestyle="-",
            marker="o",
            label="ACO best route",
            alpha=0.9,
        )

    ax.set_title("TSP Cities and Routes")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend(loc="best")

    fig.tight_layout()
    return fig


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

### How to play

1. Look at the map of cities (labeled A, B, C, ...).
2. In the text box, type the order in which you want to visit them, e.g.:
   - `A-B-C-D-E-F-G-H`
3. Click **Submit My Route** to see:
   - Your route drawn on the map.
   - Your total route length.
4. Choose ACO parameters in the sidebar (or use defaults).
5. Click **Run Ant Colony** to see:
   - The best route the ants found.
   - A chart of the best distance over iterations.
   - A percentage comparison between your route and ACO’s best.
        """
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

    with col_left:
        st.subheader("City Map & Routes")
        fig_map = plot_routes(
            cities=cities,
            labels=labels,
            user_route=st.session_state.user_route,
            aco_route=st.session_state.aco_best_route,
        )
        st.pyplot(fig_map)

    with col_right:
        st.subheader("Your Route")

        st.markdown(
            f"Cities available (in order): **{', '.join(labels)}**\n\n"
            "Enter a route that uses **each city exactly once**.\n\n"
            "Example: `A-B-C-D-E-F-G-H`"
        )

        route_input = st.text_input(
            "Type your route here:",
            value="-" .join(labels),
            key="route_input",
        )

        submit_route = st.button("Submit My Route")

        if submit_route:
            parsed_route = parse_route_input(route_input, labels)
            if parsed_route is None:
                st.error(
                    "Could not understand your route. "
                    "Please use letters separated by '-' or ',' (e.g., A-B-C-D)."
                )
            else:
                valid, msg = validate_route(parsed_route, len(labels))
                if not valid:
                    st.error(msg)
                else:
                    st.session_state.user_route = parsed_route
                    user_dist = compute_route_length(parsed_route, dist_matrix)
                    st.session_state.user_distance = user_dist
                    st.success(f"Your route length (including return to start): **{user_dist:.3f}**")

        if st.session_state.user_route is not None and st.session_state.user_distance is not None:
            st.info(
                f"Current stored route: "
                f"{' - '.join(labels[i] for i in st.session_state.user_route)}\n\n"
                f"Length: **{st.session_state.user_distance:.3f}**"
            )

        st.markdown("---")
        run_aco_clicked = st.button("Run Ant Colony")

        if run_aco_clicked:
            with st.spinner("Ants are exploring routes..."):
                aco = AntColony(
                    distance_matrix=dist_matrix,
                    num_ants=num_ants,
                    num_iterations=num_iterations,
                    alpha=alpha,
                    beta=beta,
                    evaporation_rate=evaporation_rate,
                    q=q,
                    start_city=0,
                )
                best_route, best_distance, best_history = aco.run()

            st.session_state.aco_best_route = best_route
            st.session_state.aco_best_distance = best_distance
            st.session_state.aco_best_history = best_history

            st.success("Ant Colony run completed!")

    # Results section
    st.markdown("---")
    st.header("Results & Comparison")

    c1, c2 = st.columns(2)

    with c1:
        if st.session_state.aco_best_distance is not None:
            st.subheader("Best Route Found by ACO")
            best_route_labels = [labels[i] for i in st.session_state.aco_best_route]
            st.write("Route (start city repeats at the end):")
            st.write(" → ".join(best_route_labels + [best_route_labels[0]]))
            st.write(f"Best route length: **{st.session_state.aco_best_distance:.3f}**")

        if st.session_state.aco_best_history is not None:
            fig_progress = plot_aco_progress(st.session_state.aco_best_history)
            st.pyplot(fig_progress)

    with c2:
        st.subheader("Your Route vs. ACO")

        if (
            st.session_state.user_distance is not None
            and st.session_state.aco_best_distance is not None
        ):
            user = st.session_state.user_distance
            best = st.session_state.aco_best_distance
            diff = user - best
            percent_diff = (diff / best) * 100

            if diff > 0:
                st.warning(
                    f"Your route is **{percent_diff:.1f}% longer** than the ACO best route.\n\n"
                    f"Try to rearrange the cities and get closer!"
                )
            elif diff < 0:
                st.success(
                    f"Impressive! Your route is **{-percent_diff:.1f}% shorter** "
                    f"than the best route the ants found with the current settings."
                )
            else:
                st.success("Perfect! Your route has exactly the same length as ACO's best route.")

        elif st.session_state.user_distance is None:
            st.info("Submit a route first to compare with the algorithm.")
        elif st.session_state.aco_best_distance is None:
            st.info("Run the Ant Colony algorithm to get a comparison.")

    # Educational explanation at the bottom
    st.markdown(
        """
---
### What the Ant Colony Algorithm is doing (in simple terms)

1. **Many ants, many routes**  
   In each iteration, several ants build complete routes through all cities.
   They tend to prefer:
   - edges with **more pheromone** (good experiences from previous ants)
   - and **shorter distances** (using the distance as a heuristic).

2. **Probability rule**  
   When an ant chooses the next city, the attractiveness of a move is roughly:
   \n\n
   `pheromone^alpha × (1/distance)^beta`
   \n\n
   Then these values are turned into probabilities.

3. **Evaporation**  
   After all ants finish their tours in an iteration, existing pheromone **evaporates**.
   This prevents the algorithm from getting stuck forever in old decisions.

4. **Deposit**  
   Ants then **deposit pheromone** on the edges of their routes.
   Shorter routes add **more** pheromone, so future ants are more likely to follow them.

5. **Over iterations**  
   Good edges reinforce each other, bad edges fade away.
   The best tour length usually **decreases over iterations**, which you can see in the progress chart.

Try changing:
- **Alpha**: bigger → pheromone matters more.
- **Beta**: bigger → distance (short routes) matters more.
- **Evaporation**: bigger → pheromone fades faster (more exploration).
- **Number of ants / iterations**: more → more searching, but slower.
        """
    )


if __name__ == "__main__":
    main()
