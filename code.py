import random
import math
import time
import os

# Global constants: the graph size, population size, and maximum passes for FM.
NUM_VERTICES = 500
POPULATION_SIZE = 50
MAX_FM_PASSES = 10000

# -------------------------------------------------------
# Graph reading and balanced solution generation
# -------------------------------------------------------

def read_graph_data(filename):
    """
    Reads the graph from the file.
    
    Why: The assignment requires ignoring coordinate data and extracting only the connectivity.
         We assume vertex IDs in the file are 1-based.
         
    Returns a list of neighbor lists (0-based indexing) for each vertex.
    """
    graph = [[] for _ in range(NUM_VERTICES)]
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vertex_index = int(parts[0])
            if vertex_index < 1 or vertex_index > NUM_VERTICES:
                print(f"Vertex index out of bounds: {vertex_index}")
                continue
            # Ignore the coordinate data and neighbor count token;
            # use tokens from index 3 onward for neighbor IDs.
            connected_vertices = [int(n) - 1 for n in parts[3:]]
            graph[vertex_index - 1] = connected_vertices
    print("Sample graph structure (first 5 vertices):")
    for i in range(5):
        print(f"Vertex {i}: {graph[i]}")
    return graph

def generate_random_solution():
    """
    Generates a balanced random solution.
    
    Why: The solution must have exactly NUM_VERTICES/2 zeros and ones.
         We first create the balanced list then shuffle to randomize the order.
    """
    half = NUM_VERTICES // 2
    solution = [0] * half + [1] * half
    random.shuffle(solution)
    return solution

# -------------------------------------------------------
# Fitness function and utility for gain computation
# -------------------------------------------------------

def fitness_function(solution, graph):
    """
    Computes the fitness as the number of cross–edges between partitions.
    
    Why: Each edge connecting vertices in different partitions is counted twice,
         so we divide by 2 to get the correct count.
    """
    total = 0
    for i, neighbors in enumerate(graph):
        for nb in neighbors:
            if solution[i] != solution[nb]:
                total += 1
    return total // 2

def compute_gain_for_vertex(i, solution, graph):
    """
    Computes the gain for vertex i (external edges minus internal edges).
    
    Why: This gain value tells us the benefit of moving the vertex.
         In a pair swap, the combined gain of two vertices drives our decision.
    """
    internal = 0
    external = 0
    for nb in graph[i]:
        if solution[i] == solution[nb]:
            internal += 1
        else:
            external += 1
    return external - internal

# -------------------------------------------------------
# Improved FM local search using candidate pair selection
# -------------------------------------------------------

def fm_heuristic(solution, graph):
    """
    Improved FM heuristic for local search in graph bipartitioning.
    
    Why: Rather than examining every pair (O(n^2)) for a swap, we only consider the top-k
         vertices (from each partition) ranked by their individual gains. This candidate-based
         selection greatly reduces the comparisons while still finding moves that improve the solution.
         
         Each swap exchanges one vertex from each partition – which automatically preserves the balance.
         
         Gains are fully recomputed at each pass for clarity (which is acceptable for n=500).
    """
    current_solution = solution.copy()
    passes = 0
    while passes < MAX_FM_PASSES:
        # Compute gains for every vertex in the current solution.
        gains = [compute_gain_for_vertex(i, current_solution, graph) for i in range(len(current_solution))]
        
        # Partition vertices by their current assignment.
        part0 = [i for i in range(len(current_solution)) if current_solution[i] == 0]
        part1 = [i for i in range(len(current_solution)) if current_solution[i] == 1]
        
        # Sort each partition by gain descending.
        part0.sort(key=lambda i: gains[i], reverse=True)
        part1.sort(key=lambda i: gains[i], reverse=True)
        
        # Limit search to top-k candidates from each partition.
        k = min(10, len(part0), len(part1))
        best_pair_gain = -float('inf')
        best_pair = None
        for i in part0[:k]:
            for j in part1[:k]:
                # If vertices are adjacent, the benefit is slightly reduced.
                penalty = 2 if j in graph[i] else 0
                pair_gain = gains[i] + gains[j] - penalty
                if pair_gain > best_pair_gain:
                    best_pair_gain = pair_gain
                    best_pair = (i, j)
        # If no swap provides an improvement, exit the loop.
        if best_pair is None or best_pair_gain <= 0:
            break
        # Swap the selected pair (this preserves the equal partition constraint).
        a, b = best_pair
        current_solution[a], current_solution[b] = current_solution[b], current_solution[a]
        passes += 1
    return current_solution

# -------------------------------------------------------
# Metaheuristics: MLS, ILS, and GLS
# -------------------------------------------------------

def MLS(graph, num_starts):
    """
    Multi-start Local Search.
    
    Why: By restarting local search from several random starting solutions,
         we increase the chances of finding a better local optimum.
    """
    best_solution = None
    best_fit = float('inf')
    for _ in range(num_starts):
        sol = generate_random_solution()
        local_opt = fm_heuristic(sol, graph)
        fit = fitness_function(local_opt, graph)
        if fit < best_fit:
            best_fit = fit
            best_solution = local_opt
    return best_solution

def mutate(solution, mutation_size):
    """
    Perturbs a solution by swapping bits.
    
    Why: The mutation must preserve balance. We swap an equal number of zeros and ones.
    """
    mutated = solution.copy()
    zeros = [i for i, bit in enumerate(mutated) if bit == 0]
    ones = [i for i, bit in enumerate(mutated) if bit == 1]
    random.shuffle(zeros)
    random.shuffle(ones)
    num_mutations = min(mutation_size, len(zeros), len(ones))
    for i in range(num_mutations):
        mutated[zeros[i]] = 1
        mutated[ones[i]] = 0
    return mutated

def ILS_annealing(graph, initial_solution, mutation_size):
    """
    Iterated Local Search with annealing-based acceptance.
    
    Why: By perturbing the current solution and accepting worse moves probabilistically,
         we can escape local optima. The mutation size is a key parameter.
    """
    best_solution = initial_solution.copy()
    best_fit = fitness_function(best_solution, graph)
    current_solution = initial_solution.copy()
    current_fit = best_fit
    no_improvement = 0
    max_no_improvement = 10
    temperature = 1000.0
    cooling_rate = 0.95
    while no_improvement < max_no_improvement:
        mutated = mutate(current_solution, mutation_size)
        local_opt = fm_heuristic(mutated, graph)
        fit = fitness_function(local_opt, graph)
        if fit < current_fit:
            current_solution = local_opt.copy()
            current_fit = fit
            if fit < best_fit:
                best_solution = local_opt.copy()
                best_fit = fit
                no_improvement = 0
        else:
            # Accept a worse solution with a probability based on the temperature.
            if math.exp((current_fit - fit) / temperature) > random.random():
                current_solution = local_opt.copy()
                current_fit = fit
            no_improvement += 1
        temperature *= cooling_rate
    return best_solution

def ILS(graph, initial_solution, mutation_size):
    """
    Simple Iterated Local Search.
    
    Why: An alternative ILS variant that only accepts an improvement.
         This version resets the improvement counter whenever a better solution is found.
    """
    best_solution = initial_solution.copy()
    best_fit = fitness_function(best_solution, graph)
    no_improvement = 0
    max_no_improvement = 10
    while no_improvement < max_no_improvement:
        mutated = mutate(best_solution, mutation_size)
        local_opt = fm_heuristic(mutated, graph)
        fit = fitness_function(local_opt, graph)
        if fit < best_fit:
            best_solution = local_opt.copy()
            best_fit = fit
            no_improvement = 0
        else:
            no_improvement += 1
    return best_solution

def get_hamming_distance(parent1, parent2):
    """Utility: Compute the Hamming distance between two solutions."""
    return sum(1 for a, b in zip(parent1, parent2) if a != b)

def balance_child(child):
    """
    Adjusts a child solution to restore balance.
    
    Why: Crossover may disturb the equal number of ones and zeros.
         This function randomly flips bits to meet the constraint.
    """
    target = len(child) // 2
    current_ones = sum(child)
    indices = list(range(len(child)))
    if current_ones > target:
        ones_indices = [i for i in indices if child[i] == 1]
        to_flip = random.sample(ones_indices, current_ones - target)
        for i in to_flip:
            child[i] = 0
    elif current_ones < target:
        zeros_indices = [i for i in indices if child[i] == 0]
        to_flip = random.sample(zeros_indices, target - current_ones)
        for i in to_flip:
            child[i] = 1
    return child

def crossover(parent1, parent2):
    """
    Performs uniform crossover that respects the balance constraint.
    
    Why: The crossover first checks the Hamming distance; if too high,
         one parent is inverted. Then, positions where parents agree are copied,
         and disagreeing positions are filled randomly, followed by a balance correction.
    """
    child = []
    hd = get_hamming_distance(parent1, parent2)
    if hd > len(parent1) / 2:
        parent1 = [1 - bit for bit in parent1]
    for i in range(len(parent1)):
        if parent1[i] == parent2[i]:
            child.append(parent1[i])
        else:
            child.append(random.choice([parent1[i], parent2[i]]))
    child = balance_child(child)
    return child

def GLS(graph, population_size, stopping_crit):
    """
    Genetic Local Search with steady–state replacement.
    
    Why: We start with a population of local optima (found via MLS) and then use
         selection and uniform crossover to generate offspring. After applying FM,
         the offspring competes with the worst solution.
    """
    num_starts = 5
    population = [MLS(graph, num_starts) for _ in range(population_size)]
    fitness_values = [fitness_function(sol, graph) for sol in population]
    best_fit = min(fitness_values)
    best_solution = population[fitness_values.index(best_fit)]
    generation_without_improvement = 0
    for _ in range(10000):
        if generation_without_improvement >= stopping_crit:
            break
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        child = crossover(parent1, parent2)
        child_opt = fm_heuristic(child, graph)
        fit_child = fitness_function(child_opt, graph)
        worst_fit = max(fitness_values)
        worst_index = fitness_values.index(worst_fit)
        if fit_child < worst_fit:
            population[worst_index] = child_opt.copy()
            fitness_values[worst_index] = fit_child
            if fit_child < best_fit:
                best_fit = fit_child
                best_solution = child_opt.copy()
                generation_without_improvement = 0
            else:
                generation_without_improvement += 1
        else:
            generation_without_improvement += 1
    return best_solution

# -------------------------------------------------------
# Experiment routines: Run each metaheuristic multiple times and save results.
# -------------------------------------------------------

def run_MLS_experiment_and_save_results():
    graph = read_graph_data("Graph500.txt")
    num_starts = 5
    loop_runs = 20
    solutions = []
    edges_counts = []
    computation_times = []
    for i in range(loop_runs):
        start_time = time.perf_counter()
        best_sol = MLS(graph, num_starts)
        end_time = time.perf_counter()
        solutions.append("".join(str(bit) for bit in best_sol))
        fit = fitness_function(best_sol, graph)
        edges_counts.append(fit)
        computation_times.append(end_time - start_time)
        print(f"MLS run {i} finished with fitness {fit}")
    with open("foundSolutionsMLS.txt", "w") as f:
        f.write("\n".join(solutions))
    with open("edgesCountsMLS.txt", "w") as f:
        f.write("\n".join(str(e) for e in edges_counts))
    with open("computationTimesMLS.txt", "w") as f:
        f.write("\n".join(str(t) for t in computation_times))
    print("MLS experiment results saved.")

def run_ILS_experiment_and_save_results():
    graph = read_graph_data("Graph500.txt")
    loop_runs = 20
    mutation_sizes = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100]
    num_starts = 5
    starting_solution = MLS(graph, num_starts)
    for mutation_size in mutation_sizes:
        solutions = []
        edges_counts = []
        computation_times = []
        start_time = time.perf_counter()
        for i in range(loop_runs):
            best_sol = ILS_annealing(graph, starting_solution, mutation_size)
            solutions.append("".join(str(bit) for bit in best_sol))
            fit = fitness_function(best_sol, graph)
            edges_counts.append(fit)
            print(f"ILS run {i} with mutation size {mutation_size} finished with fitness {fit}")
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / loop_runs
        computation_times = [avg_time] * loop_runs
        with open(f"ILSAnnealingFoundtrainingtimes_mutationSize{mutation_size}.txt", "w") as f:
            f.write("\n".join(str(avg_time) for _ in range(loop_runs)))
        with open(f"ILSAnnealingFoundSolutions_mutationSize{mutation_size}.txt", "w") as f:
            f.write("\n".join(solutions))
        with open(f"ILSAnnealingEdgesCounts_mutationSize{mutation_size}.txt", "w") as f:
            f.write("\n".join(str(e) for e in edges_counts))
        print(f"ILS experiment for mutation size {mutation_size} results saved.")

def run_GLS_experiment_and_save_results():
    graph = read_graph_data("Graph500.txt")
    loop_runs = 20
    stopping_crits = [1, 2]
    for stopping_crit in stopping_crits:
        solutions = []
        edges_counts = []
        computation_times = []
        start_time = time.perf_counter()
        for i in range(loop_runs):
            best_sol = GLS(graph, POPULATION_SIZE, stopping_crit)
            solutions.append("".join(str(bit) for bit in best_sol))
            fit = fitness_function(best_sol, graph)
            edges_counts.append(fit)
            print(f"GLS run {i} with stopping crit {stopping_crit} finished with fitness {fit}")
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / loop_runs
        computation_times = [avg_time] * loop_runs
        with open(f"GLSFoundtrainingtimes_stoppingcriteria{stopping_crit}.txt", "w") as f:
            f.write("\n".join(str(avg_time) for _ in range(loop_runs)))
        with open(f"GLSFoundSolutions_stoppingcriteria{stopping_crit}.txt", "w") as f:
            f.write("\n".join(solutions))
        with open(f"GLSEdgesCounts_stoppingcriteria{stopping_crit}.txt", "w") as f:
            f.write("\n".join(str(e) for e in edges_counts))
        print(f"GLS experiment for stopping criteria {stopping_crit} results saved.")

# -------------------------------------------------------
# Main execution: Run experiments sequentially.
# -------------------------------------------------------

if __name__ == '__main__':
    run_MLS_experiment_and_save_results()
    print("MLS experiments done.")
    run_ILS_experiment_and_save_results()
    print("ILS experiments done.")
    run_GLS_experiment_and_save_results()
    print("GLS experiments done.")
