
from numpy import random
from metrics.metrics import *
import torch.nn as nn
import matplotlib.pyplot as plt
from search_space import BUILDING_BLOCKS 
from metrics.utils_metrics import compute_metrics_population


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# INPUT = torch.rand(1,3,224,224)

def get_rank_based_on_metrics(exemplars, metrics, weight_params_flops=1):
    """
        Based on the indicated metrics, we assign a score to each model. 
        The score is based on the model's position in the total rank (e.g. model -> position 10 -> obtains a score = 10).
        For SynFlow and NASWOT, the model is ranked in ascending order to favor models with high SynFlow and NASWOT values. 
        For FLOPS and #Parameters, the model is ranked in descending order. This way, models with fewer parameters and FLOPS will obtain a better score. 
        For FLOPS and #Parameters, we apply a weight:

            - A weight > 1 gives more importance to FLOPS and parameters than to other metrics.
            - A weight < 1 gives more importance to SynFlow and NASWOT scores than to FLOPS and parameters.
            - weight = 1 gives same importance
    
    """

    scores = {exemplar : 0 for exemplar in exemplars}

    for metric in metrics:

        if metric == "#Parameters" or metric == "FLOPS":
            rank_metric = sorted(exemplars, key=lambda x: x.get_metric_score(metric), reverse=True)

            for i,exemplar in enumerate(rank_metric):
                scores[exemplar] += i*weight_params_flops
        else:
            rank_metric = sorted(exemplars, key=lambda x:  x.get_metric_score(metric))

            for i,exemplar in enumerate(rank_metric):
                scores[exemplar] += i

    final_rank = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)).keys()

    return list(final_rank)



def get_top_k_models(networks, k):
    return networks[:k]



channels = [16, 32, 64, 96, 160, 320]
kernel_sizes = [3,5,7]
expansion_factors = [2,4,6]  # for only inverted residual block
strides = [1,2]

## Structure of block:
## [ block_type, output_channels, kernel, stride, expansion_factor ]   

def generate_random_network_encode(input_channels_first, num_max_blocks, fixed_size):

    #input_channels = input_channels_first
    blocks = []

    for _ in range(random.randint(1,num_max_blocks) if fixed_size == False else num_max_blocks):

        block = generate_random_block()
        blocks.append(block)
       
       
    return blocks    
        
def generate_random_block():

    block_type = random.choice(list(BUILDING_BLOCKS.keys()))
    block = generate_random_params(block_type)
    return block 


def generate_random_params(block_type):
     
    kernel_size = 0
    stride = 0
    expansion_factor = 0

    # if block_type == "ConvNeXt":
    #     output_channels = input_channels
    # else:
    #     output_channels = random.choice(channels)
    #     kernel_size = random.choice(kernel_sizes)
    #     stride = random.choice(strides, p=[0.75, 0.25])

    output_channels = random.choice(channels)
    kernel_size = random.choice(kernel_sizes)
    stride = random.choice(strides, p=[0.6, 0.4])
    if block_type == "InvertedResidual" or block_type == "ConvNeXt":
        expansion_factor = random.choice(expansion_factors)
        
    
    return [block_type, output_channels, kernel_size,stride,expansion_factor]
    


def plot_metrics(exemplars):

    synflow_socres = [exemplar.metrics["synflow"] for exemplar in exemplars]
    naswot_scores = [exemplar.metrics["naswot"] for exemplar in exemplars]

    fig, ax = plt.subplots()
    ax.plot(naswot_scores, synflow_socres)
    ax.set_title("Synflow score vs NASWOT score")
    ax.set_xlabel("NASWOT score")
    ax.set_ylabel("Synflow score")
    plt.show()
    
    return 
