from argparse import ArgumentParser
from models.MobileNetV2 import MobileNetV2
from train import trainer
import torch
from random_search import random_search
from dataset import get_data_loader 
from fvcore.nn import FlopCountAnalysis, flop_count_table
from metrics.metrics import *
from utils import get_ranking_based_on_metrics, get_top_k_models, isfeasible

def main():
    #parser = ArgumentParser()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # batch size = 64, 96, 128 (possible error CUDA out of memory)
    
    
    #model = MobileNetV2(width_mult=0.35).to(device)




    networks = random_search(num_iterations=100,num_max_blocks=5)


    #input = torch.rand(1,3,96,96).type(torch.float16).to(device)
    # image resolution: [96, 128, 160, 192, 224]
    input = torch.rand(1,3,224,224).to(device)

    feasible_networks = [model for model in networks if isfeasible(model,input,max_flops=200*(10**6), max_params=25*(10**5))]
    
    print(len(feasible_networks))
    """
    for i,model in enumerate(feasible_networks):
        print(f"Model {i}:")
        print(model)
        print(f"#Parameters: {get_params(model)}")
        print(f"#Flops: {count_flops(model,input)}")
        """
    #result = networks[0](input)
    #print(result)
    #model = networks[0]
    #print(model)
    #flops = FlopCountAnalysis(model, input)
    
    #print(flop_count_table(flops))
    
    """
    

                                                                        
    """

    """
    for i,model in enumerate(networks):
        #print(f"Model 1:\n {model}")
        print(f"Model {i}:")
        print(f"#Parameters: {get_params(model)}")
        print(f"#Flops: {count_flops(model,input)}")
        print(f"Naswot score: {compute_naswot_score(model,input,device)}")
        print(f"Synflow score: {compute_synflow_per_weight(model,input,device)}")
    """

    feasible_networks_ranked = get_ranking_based_on_metrics(feasible_networks, input, device)
    best_models = get_top_k_models(feasible_networks_ranked, k=3)
    
    for model in best_models:
        print(model)
        print(f"#Parameters: {get_params(model)}")
        print(f"#Flops: {count_flops(model,input)}")
        print(f"Naswot score: {compute_naswot_score(model,input,device)}")
        print(f"Synflow score: {compute_synflow_per_weight(model,input,device)}")
    
    

    """

    root_data = "COCOdataset/all2017"
    path_annotations_train = "visualwakewords/instances_train.json"
    path_annotations_val = "visualwakewords/instances_val.json"

    train_dataloader, val_dataloader, test_dataloader = get_data_loader(root_data,
                                                                        path_annotations_train,
                                                                        path_annotations_val,
                                                                        128)
    
    MV2_val_loss, MV2_val_accuracy, MV2_train_loss, MV2_train_accuracy = trainer(train_dataloader,
                                                                                 val_dataloader,
                                                                                 test_dataloader,
                                                                                 learning_rate=0.1,
                                                                                 weight_decay=0.000001,
                                                                                 momentum=0.9,
                                                                                 epochs=2,
                                                                                 model= model,
                                                                                 device=device )
    
                                                                                 """

    return 






if __name__ == '__main__':
    main()