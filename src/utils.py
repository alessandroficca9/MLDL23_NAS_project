
from numpy import random
from metrics.metrics import *
import torch.nn as nn

from search_space import BUILDING_BLOCKS 
from metrics.utils_metrics import compute_metrics_population


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# INPUT = torch.rand(1,3,224,224)

def get_rank_based_on_metrics(exemplars, metrics):

    scores = {exemplar : 0 for exemplar in exemplars}

    for metric in metrics:
        rank_metric = sorted(exemplars, key=lambda x:  x.get_metric_score(metric), reverse=True)

        for i,exemplar in enumerate(rank_metric):
            scores[exemplar] += i

    final_rank = dict(sorted(scores.items(), key=lambda x: x[1])).keys()

    return list(final_rank)



def get_top_k_models(networks, k):
    return networks[:k]



channels = [16, 32, 64, 96, 160, 320]
kernel_sizes = [3,5,7]
expansion_factors = [2,4,6]  # for only inverted residual block
strides = [1,2]

## Structure of block:
## [ block_type, output_channels, kernel, stride, expansion_factor ]   

def generate_random_network_encode(input_channels_first, num_max_blocks):

    #input_channels = input_channels_first
    blocks = []

    for _ in range(random.randint(1,num_max_blocks)):

        block = generate_random_block()
        blocks.append(block)
        #input_channels = block[1]   # next input channels = current output channels
       
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
    stride = random.choice(strides, p=[0.7, 0.3])
    if block_type == "InvertedResidual" or block_type == "ConvNeXt":
        expansion_factor = random.choice(expansion_factors)
        
    
    return [block_type, output_channels, kernel_size,stride,expansion_factor]
    
