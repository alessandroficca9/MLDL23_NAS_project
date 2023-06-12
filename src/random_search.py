from search_space import *
import random
import math 
from exemplar import Exemplar
from utils import generate_random_network_encode, get_rank_based_on_metrics, get_top_k_models
from metrics.utils_metrics import  isfeasible, compute_metrics
from tqdm import tqdm

def search_random(num_iterations, num_max_blocks, max_params, max_flops, input_channels_first, \
                   k, metrics,weigth_params_flops, inputs, device, fixed_size=False):

    print("Start random search ...")
    population = []

   
    for i in tqdm(range(num_iterations)):
        #print(f"Iteration: {i}/{num_iterations}")
            
        network_encoded = generate_random_network_encode(input_channels_first=input_channels_first, num_max_blocks=num_max_blocks, fixed_size=fixed_size)
            
        exemplar = Exemplar(network_encoded)

        if isfeasible(exemplar, max_params=max_params, max_flops=max_flops, inputs=inputs, device=device):
            compute_metrics(exemplar, inputs, device)
            exemplar.model = None
            population.append(exemplar)
        else:
            del exemplar
             
    
    print("Finish random search.")
    print(f"Remaining {len(population)} that satisfy constraints")
    population_rank = get_rank_based_on_metrics(population, metrics,weight_params_flops=weigth_params_flops)
    #top_k_models = get_top_k_models(population_rank, k)

   

    #return top_k_models
    return population_rank

    

        

