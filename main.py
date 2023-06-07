import random
import subprocess
import math
import time
import numpy as np
from multiprocessing import Pool

# This function generates a random set of cities in a text file
def generate_input(file_name, num_cities):
    with open(file_name, 'w') as file:
        for i in range(num_cities):
            x = random.randint(0, 10000)
            y = random.randint(0, 10000)
            file.write(f"{i} {x} {y}\n")

# This function reads the input file and constructs a dictionary of cities
def read_input(file_name):
    cities = {}
    with open(file_name, 'r') as file:
        for line in file.readlines():
            city_info = line.split()
            city_id = int(city_info[0])
            x = int(city_info[1])
            y = int(city_info[2])
            cities[city_id] = (x, y)
    return cities

# This function computes the distance between all pairs of cities and returns a 2D numpy array of distances
def compute_distances(cities):
    n = len(cities)
    distances = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = distance(cities[i], cities[j])
    return distances

# This function uses the Nearest Neighbor heuristic to generate an initial tour
def nearest_neighbor_global(distances):
    current_city = np.random.choice(len(distances))
    tour = [current_city]
    unvisited_cities = set(range(len(distances)))
    unvisited_cities.remove(current_city)

    while len(tour) < math.ceil(len(distances) / 2):
        next_city = min(unvisited_cities, key=lambda city: distances[current_city][city])
        unvisited_cities.remove(next_city)
        tour.append(next_city)
        current_city = next_city

    return tour

# This function optimizes the tour using the 2-opt swap heuristic
def two_opt_global(tour, distances):
    n = len(tour)
    max_iter = 1000  # Set a maximum number of iterations

    for _ in range(max_iter):
        i, j = random.randint(0, n - 3), random.randint(2, n - 1)
        if j - i == 1: continue  # If the two cities are consecutive in the tour, skip
        new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
        new_distance = total_distance(new_tour, distances)
        old_distance = total_distance(tour, distances)
        if new_distance < old_distance:
            tour = new_tour

    return tour

# This function writes the tour to an output file
def write_output(file_name, tour, total_distance):
    with open(file_name, 'w') as file:
        file.write(str(total_distance) + '\n')
        for city in tour:
            file.write(str(city) + '\n')
        file.write('\n')

# This function verifies the solution using a verifier script
def verify_solution(instance_file, solution_file):
    verifier_script = 'half_tsp_verifier.py'
    result = subprocess.run(['python', verifier_script, instance_file, solution_file], capture_output=True, text=True)
    print("Output:")
    print(result.stdout)
    print("Errors:")
    print(result.stderr)

# This function computes the Euclidean distance between two cities
def distance(c1, c2):
    return round(math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

# This function computes the total distance of a tour
def total_distance(tour, distances):
    return sum(distances[tour[i-1]][tour[i]] for i in range(1, len(tour))) + distances[tour[-1]][tour[0]]

if __name__ == "__main__":
    input_files = ['test-input-1.txt', 'test-input-2.txt', 'test-input-3.txt', 'test-input-4.txt']
    output_files = ['test-output-1.txt', 'test-output-2.txt', 'test-output-3.txt', 'test-output-4.txt']

    # For each input file, read the cities, compute the distances, generate an initial tour using the Nearest Neighbor heuristic,
    # optimize the tour using the 2-opt swap heuristic, and write the tour to an output file
    for input_file, output_file in zip(input_files, output_files):
        start_time = time.time()

        cities = read_input(input_file)
        distances = compute_distances(cities)
        tour = []

        with Pool() as p:
            tour = p.apply(nearest_neighbor_global, args=(distances,))
            tour = p.apply(two_opt_global, args=(tour, distances))

        dist = total_distance(tour, distances)
        print(f"Distance Traveled: {dist}")
        write_output(output_file, tour, dist)
        verify_solution(input_file, output_file)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time} seconds")
