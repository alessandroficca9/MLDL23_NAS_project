
from exemplar import Exemplar
from random_search import generate_random_network_encode, get_rank_based_on_metrics, get_top_k_models
import random
from metrics.utils_metrics import compute_metrics_population

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

    # pruning first population

    for step in range(max_step):
        
        sampled = random.sample(population, n=5)
        sampled = get_rank_based_on_metrics(sampled, metrics)

        parents = get_top_k_models(sampled, k=2)

        # add the children
        for child in mutation(parents):
            population.append(child)

        ## pruninig
        # remove non feasible elements
        # kill the oldest
        # keep the top N models

        
        

        


def mutation(parents, crossover=True, mutation_rate=0.5):

    while True:
        child_1 = parents[0].mutate()

    child_1 = parents[0].mutate()
    child_2 = parents[1].mutate()
    
    if crossover:
        child_3 = crossover(parents[0], parents[1])
        return child_1, child_2, child_3
    
    return child_1, child_2

    