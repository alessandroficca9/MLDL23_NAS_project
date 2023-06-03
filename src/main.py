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
    
    

    
    # root_data = "COCOdataset/all2017"
    # path_annotations_train = "MLDL23_NAS_Project/visualwakewords/instances_train.json"
    # path_annotations_val = "MLDL23_NAS_Project/visualwakewords/instances_val.json"

    # train_dataloader, val_dataloader, test_dataloader = get_data_loader(root_data,
    #                                                                     path_annotations_train,
    #                                                                     path_annotations_val,
    #                                                                     64)
    # inputs = next(iter(train_dataloader))
    
    inputs = torch.rand(4,3,224,224)
    
    # best_models = search_random(num_iterations=2,
    #                             num_max_blocks=7,
    #                             max_params= 25*(10**5),
    #                             max_flops= 200*(10**6), 
    #                             input_channels_first=3,
    #                             k=3,
    #                             metrics= ["synflow", "naswot"],
    #                             inputs=inputs,
    #                             device=device
    #                             )
    
  
    # print("best models random")
    # if len(best_models) > 0:
    #     for model in best_models:
    #         print(model.get_model())
    #         print(model.get_cost_info())
    #         print(model.get_metric_score("synflow"))
    #         print(model.get_metric_score("naswot"))
        

    best_models_ea = search_evolution(population_size=5,
                 num_max_blocks=3,
                 max_step=5,
                 metrics=['synflow', 'naswot'],
                 inputs = inputs,
                 device=device,
                 max_flops=200*(10**6),
                 max_params=25*(10**5))

    
    print("best models ea")
    if len(best_models_ea) > 0:
        for model in best_models_ea:
            print(model.get_model())
            print(model.get_cost_info())
            print(model.get_metric_score("synflow"))
            print(model.get_metric_score("naswot"))

    

    
    
    # MV2_val_loss, MV2_val_accuracy, MV2_train_loss, MV2_train_accuracy = trainer(train_dataloader,
    #                                                                              val_dataloader,
    #                                                                              test_dataloader,
    #                                                                              learning_rate=0.1,
    #                                                                              weight_decay=0.000001,
    #                                                                              momentum=0.9,
    #                                                                              epochs=2,
    #                                                                              model= model,
    #                                                                              device=device )
    
                                                                                 

    return 






if __name__ == '__main__':
    main()