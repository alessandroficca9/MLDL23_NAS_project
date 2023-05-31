from argparse import ArgumentParser
from models.MobileNetV2 import MobileNetV2
from train import trainer
import torch
from random_search import search_random
from dataset import get_data_loader 
from evolution_search import search_evolution



def main():
    #parser = ArgumentParser()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # batch size = 64, 96, 128 (possible error CUDA out of memory)
    
    
    #model = MobileNetV2(width_mult=0.35).to(device)

    
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
    root_data = "COCOdataset/all2017"
    path_annotations_train = "visualwakewords/instances_train.json"
    path_annotations_val = "visualwakewords/instances_val.json"

    train_dataloader, val_dataloader, test_dataloader = get_data_loader(root_data,
                                                                        path_annotations_train,
                                                                        path_annotations_val,
                                                                        64)
    inputs = next(iter(train_dataloader))
    #print(inputs)

    # best_models = search_random(num_iterations=15,
    #                             num_max_blocks=7,
    #                             max_params= 25*(10**5),
    #                             max_flops= 200*(10**6), 
    #                             input_channels_first=3,
    #                             k=3,
    #                             metrics= ["synflow", "naswot"],
    #                             inputs=inputs,
    #                             device=device
    #                             )
    
  
    # print(len(best_models))
    # if len(best_models) > 0:
    #     for model in best_models:
    #         print(model.get_model())
    #         print(model.get_cost_info())
    #         print(model.get_metric_score("synflow"))
    #         print(model.get_metric_score("naswot"))
        

    search_evolution(population_size=25,
                 num_max_blocks=9,
                 max_step=100,
                 metrics=['synflow', 'naswot'],
                 inputs = inputs,
                 device=device,
                 max_flops=200*(10**6),
                 max_params=25*(10**5))

    """

    

    
    
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