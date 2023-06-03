from argparse import ArgumentParser
from models.MobileNetV2 import MobileNetV2
from train import trainer
import torch
from random_search import search_random
from dataset import get_data_loader 
from evolution_search import search_evolution
import argparse 



def main():
    
    parser = ArgumentParser()
    
    parser.add_argument("--algo", type=str, default='ea_search', choices=("random_searc", "ea_search"))
    parser.add_argument('--max_flops', type=float, default=float('inf'))
    parser.add_argument('--max_params', type=float, default=float('inf'))
    parser.add_argument('--n_random', type=int, default=100)
    parser.add_argument('--initial_pop', type=int, default=25)
    parser.add_argument('--generation_ea', type=int, default=100)
    parser.add_argument("--save", type=bool, default=False)

    # parser.add_argument("--root_data", type=str, default="COCOdataset/all2017")
    # parser.add_argument("--ann_train", type=str, default="MLDL23_NAS_Project/visualwakewords/instances_train.json")
    # parser.add_argument("--ann_val", type=str, default="MLDL23_NAS_Project/visualwakewords/instances_val.json")
    # parser.add_argument("--batch_size", type=int, default=64)

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
    
    mini_batch_size = 8
    inputs = torch.rand(mini_batch_size,3,224,224)

    args = parser.parse_args()
    max_flops = args.max_flops
    max_params = args.max_params
    
    if args.algo == "ea_search":
        population_size = args.initial_pop
        num_generations = args.generation_ea
        best_models = search_evolution(population_size=population_size,
                                       num_max_blocks=7,
                                       max_step=num_generations,
                                       metrics=['synflow', 'naswot'],
                                       inputs=inputs,
                                       device=device,
                                       max_flops=max_flops,
                                       max_params=max_params)
    
    elif args.algo == "random_search":
        num_models = args.n_random
        best_models = search_random(num_iterations=num_models,
                                    num_max_blocks=3,
                                    max_params=max_params,
                                    max_flops=max_flops,
                                    input_channels_first=3,
                                    k=3,
                                    metrics=['synflow','naswot'],
                                    inputs=inputs,
                                    device=device)
    

    print("best models ea")
    if len(best_models) > 0:
        for model in best_models:
            print(f"model: {model.get_model()}")
            print(f"info flops and params {model.get_cost_info()}")
            print(f"synflow score: {model.get_metric_score('synflow')}")
            print(f"naswot score: {model.get_metric_score('naswot')}")


    
    if args.save:
        model = best_models[0]
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save('model_scripted.pt') # Save
    
    
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