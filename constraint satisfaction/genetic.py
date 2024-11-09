import random
import math

def n_satisfied(state):
    """Returns the number of satisfied constraints

    Args:
        state (array[int]): Array of variable assignments in the format [A, B, C, D, E, F, G, H]

    Returns:
        int: Number of satisfied constraints
    """
    n_s = 0
    
    A, B, C, D, E, F, G, H = (var for var in state)
    if (A > G):
        n_s += 1
    if (A <= H):
        n_s += 1
    if (abs(F - B) == 1):
        n_s += 1
    if (G < H):
        n_s += 1
    if (abs(G - C) == 1):
        n_s += 1
    if ((H - C) % 2 == 0):
        n_s += 1
    if (H != D):
        n_s += 1
    if (D >= G):
        n_s += 1
    if (D != C):
        n_s += 1
    if (E != C):
        n_s += 1
    if (E < D - 1):
        n_s += 1
    if (E != H - 2):
        n_s += 1
    if (G != F):
        n_s += 1
    if (H != F):
        n_s += 1
    if (C != F):
        n_s += 1
    if (D != F - 1):
        n_s += 1
    if (abs(E - F) % 2 == 1):
        n_s += 1
        
    return n_s

def get_survival_ranges(population, fitness_fn):
    """Returns an array with each state given a range of floats with length proportional to its fitness score which can later be sampled 

    Args:
        population (array[array[int]]): 2-dimensional array of states, each state being an array of variable assignments
        fitness_fn (function): Fitness function used to evaluate the states

    Returns:
        array: array of ranges with the i-th variable having range [arr[i - 1], arr[i]] (arr[i - 1] is 0 when i = 0)
    """
    s_arr = [ ]
    
    for i in range(len(population)):
        n_s = fitness_fn(population[i])
        s_arr.append(n_s)
        
    s = sum(s_arr)    
    for i in range(len(s_arr) - 1):
        print(f"State {i + 1}: {population[i]}\n" +
              f"         Fitness score: {s_arr[i]}, " + 
              f"Parent likelihood: {math.floor((s_arr[i] / s) * 10000)/100}%")
        s_arr[i] /= s
        if (i != 0):
            s_arr[i] += s_arr[i - 1]        
    
    return s_arr

def select_parents_crossover(survival_range, num_pairs):
    """Select which parents to cross and a crossover point based off of samples from a uniform probability distribution 

    Args:
        survival_range (array[int]): Array of ranges corresponding to each state's survival
        num_pairs (int): Number of parent pairs to generate

    Returns:
        array[array[int]]: 2-dimensional array with each entry in the format [parent1, parent2, crossover_point]
    """
    parent_pairs = []
    
    for i in range(num_pairs):
        p1 = random.uniform(0, 1)
        p2 = random.uniform(0, 1)

        for j in range(len(survival_range)):
            if (p1 <= survival_range[j]):
                p1 = j
                break

        for j in range(len(survival_range)):
            if (p2 <= survival_range[j]):
                p2 = j
                break
            
        crossover_point = random.randint(0, len(survival_range) - 1)

        parent_pairs.append([p1, p2, crossover_point])
    
    return parent_pairs

def genetic(init_pop, var_domain, fitness_fn, num_iters, max_c_satisfy):
    """Genetic algorithm to find solutions of CSPs

    Args:
        init_pop (array[array[int]]): 2-dimensional array of states, each state being an array of variable assignments
        var_domain (array[int]): Domain of the variables
        fitness_fn (function): Fitness function used to evaluate states
        num_iters (int): Number of iterations to run the algorithm for
        max_c_satisfy (int): Maximum number of constraints that can be satisfied (goal)

    Returns:
        array[array[int]]: 2-dimensional array of states after running the algorithm
    """
    pop_ = init_pop.copy()
    
    # Lookup table to convert indices to variable names
    int_to_var = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
    
    # Genetic algorithm for-loop
    for i in range(num_iters):
        print(f"Generation {i}:")
        
        # Getting fitness scores and corresponding survival ranges
        sr = get_survival_ranges(pop_, fitness_fn)
        
        # Getting parent pairs and a random crossover point for each pair (Selection)
        parent_pairs = select_parents_crossover(sr, int(len(pop_) / 2))
        
        # Initializing an empty array for the next generation
        new_gen = []
        
        print("\nCrossing")
        # Performing crossing for each of the parent pairs about the crossover points (Crossing)
        for j in range(len(parent_pairs)):
            parent_pair = parent_pairs[j]
            
            print(f"Pairing {j + 1}: " +
                  f"State {parent_pairs[j][0] + 1} x State {parent_pairs[j][1] + 1} " +
                  f"with crossover point {parent_pairs[j][2]}")
            
            c1 = pop_[parent_pair[0]].copy()
            c2 = pop_[parent_pair[1]].copy()
            crossover_point = parent_pair[2]
            
            temp = c1[crossover_point: ]
            
            c1[crossover_point: ] = c2[crossover_point: ]
            c2[crossover_point: ] = temp
            
            print(f"Children: {c1} \n" + 
                  f"          {c2}")
            
            new_gen.append(c1)
            new_gen.append(c2)
            
        print("\nMutations")
        # Performing mutation with a 30% chance of a variable in a state being mutated (Mutation)
        for j in range(len(new_gen)):
            curr_c = new_gen[j]
            
            if (random.uniform(0, 1) <= 0.3):
                mutation_var_i = random.randint(0, len(c1) - 1)
                mutated_var = var_domain[random.randint(0, len(var_domain) - 1)]
                
                curr_c[mutation_var_i] = mutated_var

                print(f"Child {j + 1} :"+
                      f"Variable {int_to_var[mutation_var_i]} "+ 
                      f"(index {mutation_var_i}) "+
                      f"mutated to {curr_c[mutation_var_i]}\n"+
                      f"        {curr_c}")
            else:
                print(f"Child {j + 1}: No mutation")
                
        print(f"\nNext generation")
        
        for j in range(len(new_gen)):
             print(f"State {j + 1}: {new_gen[j]}")
        
        # Updating population to be the new generation
        pop_ = new_gen
        
        # Checking to see if any state satisfies all constraints
        # If yes, shout that state and break
        for j in range(len(pop_)):
            if (fitness_fn(pop_[i]) == max_c_satisfy):
                print(f"\nFound solution: {pop_[i]}")
                return pop_

        print("\n" + "=" * 51 + "\n")
    
    return pop_

# Example
init_pop = [[1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3],
            [4,4,4,4,4,4,4,4],
            [1,2,3,4,1,2,3,4],
            [4,3,2,1,4,3,2,1],
            [1,2,1,2,1,2,1,2],
            [3,4,3,4,3,4,3,4]]

new_pop = genetic(init_pop, [1, 2, 3, 4], n_satisfied, 4, 17)

