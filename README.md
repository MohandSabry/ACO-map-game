#**Interactive Ant Colony TSP Learning Game**

This project is an interactive, browser-based learning game that teaches the Traveling Salesman Problem (TSP) and Ant Colony Optimization (ACO).
It’s built in pure Python using Streamlit.

With the app, learners can:
	•	Explore a small 2D map of cities.
	•	Enter their own proposed TSP route.
	•	Run an Ant Colony algorithm on the same map.
	•	Compare their route against the algorithm’s best result and watch how it improves over time.

⸻

1. Project Structure
	•	app.py
The main Streamlit interface. It:
	•	Displays the city map and all routes.
	•	Accepts a user-entered route (by city labels).
	•	Runs the ACO algorithm, then shows the results and explanations.
	•	aco.py
The core Ant Colony Optimization implementation for the TSP. It includes:
	•	A pheromone matrix
	•	A distance-based heuristic (1 / distance)
	•	The probabilistic transition rule for choosing the next city
	•	Evaporation and pheromone-update logic
	•	tsp_instance.py
Handles generating and labeling cities:
	•	generate_cities() creates random 2D coordinates.
	•	get_city_labels() assigns labels A, B, C, …
	•	utils.py
General helper functions:
	•	Build the distance matrix
	•	Compute the length of a given route
	•	Parse and validate user-submitted routes
	•	requirements.txt
Lists all Python dependencies.

⸻

2. How the Game Works
	1.	The app generates a set of cities inside a unit square.
	2.	The cities are plotted and labeled (A, B, C, …).
	3.	The user enters a route, such as: A-B-C-D-E-F-G-H.
The app:
	•	Parses the labels into indices
	•	Checks that each city appears exactly once
	•	Computes the full route length (including the return to the starting city)
	4.	The user can adjust ACO settings or stick with defaults:
	•	Number of ants
	•	Number of iterations
	•	Alpha (weight of pheromone)
	•	Beta (weight of distance)
	•	Evaporation rate
	•	Pheromone deposit factor Q
	5.	The AntColony algorithm runs:
	•	Each ant builds a complete route
	•	Pheromone evaporates and is reinforced based on route quality
	•	The best route so far is tracked across iterations
	6.	The app then:
	•	Draws the ACO best route on the map
	•	Shows a line chart of best route length over time
	•	Compares the user’s route to the algorithm’s result as a percentage difference
