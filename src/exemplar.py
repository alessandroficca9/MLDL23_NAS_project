
from copy import deepcopy
import torch
from search_space import NetworkDecoded
from utils import generate_random_block, generate_random_params
import random
import numpy as np

class Exemplar:

    def __init__(self, network_encode, age=0):         # add other attributes for evolution
        super().__init__()

        self.network_encode = network_encode
        self.metrics = None
        #self.cost_info = None
        self.model = None 
        self.age = 0

    def get_metric_score(self, metric):
        
        return self.metrics[metric]

    def get_model(self):
        
        if self.model == None:
            self.model = NetworkDecoded(self.network_encode, num_classes=2)
        
        return self.model
    
    def get_cost_info(self):
        params = self.metrics["#Parameters"]
        flops = self.metrics["FLOPS"]
        return params, flops
    
    def mutate(self, random=True):
        
        ## Mutation_options = ["Change a block", "Change params of a block", "add a block"]
        probs = [2/6, 3/6, 1/6]
        chosen = np.random.multinomial(n=1, pvals=probs)
        chosen = np.argmax(chosen)
        
        # choose random idx of block
        if random:
            idx_block = np.random.randint(0, len(self.network_encode))
            if idx_block == 0:
                input_channels = 3
            else:
                input_channels = self.network_encode[idx_block-1][1]

        new_network_encode = deepcopy(self.network_encode)

        if chosen == 0:     # Change a block    
            new_network_encode[idx_block] = generate_random_block(input_channels=input_channels)  

        elif chosen == 1:   #change params of a block
            if self.network_encode[idx_block][0] != 'ConvNeXt':
                new_network_encode[idx_block] = generate_random_params(block_type= self.network_encode[idx_block][0],
                                                                        input_channels= input_channels)
        else:   # add a block
            input_channels = self.network_encode[-1][1]
            new_network_encode.append( 
                generate_random_block(input_channels=input_channels)
            )

        return Exemplar(new_network_encode, age=0)
    
    def set_age(self,age):
        self.age = age 
    
    
        
        


    