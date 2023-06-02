
from exemplar import Exemplar
from random_search import generate_random_network_encode, get_rank_based_on_metrics, get_top_k_models
import random
from metrics.utils_metrics import compute_metrics_population
from ea_utils import update_history, prune_population, clean_history

def population_init(N, num_max_blocks):

    population = []

    for _ in range(N):
        network_encode = generate_random_network_encode(input_channels_first=3, num_max_blocks=num_max_blocks)
        exemplar = Exemplar(network_encode, age=0)
        population.append(exemplar)
    
    return population



def search_evolution(population_size, num_max_blocks, max_step, metrics, inputs, device, max_flops, max_params):

    population = population_init(population_size, num_max_blocks)
    compute_metrics_population(population, inputs, device)
    #compute_metrics_population(population, inputs, device)
    history = {}
    history = update_history(population, history)

    # pruning first generation
    population = prune_population(population, inputs, device, kill_oldest=False, top_N=population_size, metrics=metrics,
                                   max_params=max_params, max_flops=max_flops)

    for step in range(max_step):
        

        sampled = random.sample(population, n=5)
        sampled = get_rank_based_on_metrics(sampled, metrics)

        parents = get_top_k_models(sampled, k=2)

        # add the children
        for child in mutation(parents, crossover=True, age=step+1):
            population.append(child)

        compute_metrics_population(population, inputs, device)
        history = update_history(population, history)

        population = prune_population(population, inputs, device, kill_oldest=True, top_N=population_size, metrics=metrics, 
                                      max_params=max_params, max_flops=max_flops)
        
    history = clean_history(history, inputs, device, max_params=max_params, max_flops=max_flops)
    final_models = get_rank_based_on_metrics(history.values(), metrics)
    best_models = get_top_k_models(final_models, k=3)
    return best_models

def mutation(parents, crossover=True, age=0):

    child_1 = parents[0].mutate()
    child_1.set_age(age)
    child_2 = parents[1].mutate()
    child_2.set_age(age)
    
    if crossover:
        child_3 = crossover(parents[0], parents[1])
        child_3.set_age(age)
        return child_1, child_2, child_3
    
    return child_1, child_2

def crossover(parent_1, parent_2):

    #different kind of crossover

    #1. for idx of block, choose randomly between the parents
    #2. decide a crossover point and join the different blocks
    network_crossover = []
    min_lenghts = min(len(parent_1.network_encode), len(parent_2.network_encode))

    if random.rand() < 0.5:

        for i in range(min_lenghts):
            block = random.choice( (parent_1.network_encode[i], parent_2.network_encode[i]))
            network_crossover.append(block)

    else:

        crossover_point = random.randint(0, min_lenghts-1)
        network_crossover = parent_1.network_encode[:crossover_point] + parent_2.network_encode[crossover_point:]
        
    
    return Exemplar(network_encode=network_crossover)


    
