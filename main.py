import random
import subprocess
import math
import time
import numpy as np
from multiprocessing import Pool

def generate_input(file_name, num_cities):
    with open(file_name, 'w') as file:
        for i in range(num_cities):
            x = random.randint(0, 10000)
            y = random.randint(0, 10000)
            file.write(f"{i} {x} {y}\n")

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

def compute_distances(cities):
    n = len(cities)
    distances = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = distance(cities[i], cities[j])
    return distances

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

# def two_opt_global(tour, distances):
#     n = len(tour)
#     improved = True
#
#     while improved:
#         improved = False
#         for i in range(n):
#             for j in range(i + 2, (n + i - 1) % n + 1):
#                 if j - i == 1: continue  # changes nothing, skip then
#
#                 old_distance = distances[tour[i-1]][tour[i]] + distances[tour[j-1]][tour[j]]
#                 new_distance = distances[tour[i-1]][tour[j-1]] + distances[tour[i]][tour[j]]
#
#                 if new_distance < old_distance:
#                     tour[i:j] = reversed(tour[i:j])
#                     improved = True
#     return tour

def two_opt_global(tour, distances):
    n = len(tour)
    max_iter = 1000  # Choose a reasonable number of iterations.

    for _ in range(max_iter):
        i, j = random.randint(0, n - 3), random.randint(2, n - 1)
        if j - i == 1: continue  # changes nothing, skip then
        new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
        new_distance = total_distance(new_tour, distances)
        old_distance = total_distance(tour, distances)
        if new_distance < old_distance:
            tour = new_tour

    return tour

def write_output(file_name, tour, total_distance):
    with open(file_name, 'w') as file:
        file.write(str(total_distance) + '\n')
        for city in tour:
            file.write(str(city) + '\n')
        file.write('\n')

def verify_solution(instance_file, solution_file):
    verifier_script = 'half_tsp_verifier.py'
    result = subprocess.run(['python', verifier_script, instance_file, solution_file], capture_output=True, text=True)
    print("Output:")
    print(result.stdout)
    print("Errors:")
    print(result.stderr)

def distance(c1, c2):
    return round(math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

def total_distance(tour, distances):
    return sum(distances[tour[i-1]][tour[i]] for i in range(1, len(tour))) + distances[tour[-1]][tour[0]]


if __name__ == "__main__":
    input_file_name = 'test-input-3.txt'
    output_file_name = 'output.txt'

    start_time = time.time()

    cities = read_input(input_file_name)
    distances = compute_distances(cities)
    tour = []

    with Pool() as p:
        tour = p.apply(nearest_neighbor_global, args=(distances,))
        tour = p.apply(two_opt_global, args=(tour, distances))

    dist = total_distance(tour, distances)
    print(f"Distance Traveled: {dist}")
    write_output(output_file_name, tour, dist)
    verify_solution(input_file_name, output_file_name)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")