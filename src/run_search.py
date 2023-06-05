from argparse import ArgumentParser
import torch
from random_search import search_random
from evolution_search import search_evolution
from models.resnet import ResNet
from models.MobileNetV2 import MobileNetV2


def main():
    
    parser = ArgumentParser()
    
    parser.add_argument("--algo", type=str, default='random_search', choices=("random_search", "ea_search","our_cnn"))
    parser.add_argument('--max_flops', type=float, default=200*(10**6))
    parser.add_argument('--max_params', type=float, default=25*(10**5))
    parser.add_argument('--n_random', type=int, default=10)
    parser.add_argument('--initial_pop', type=int, default=25)
    parser.add_argument('--generation_ea', type=int, default=100)
    parser.add_argument("--save", type=bool, default=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    mini_batch_size = 4
    if device == "cpu":
         inputs = torch.rand(mini_batch_size,3,224,224).to(device)
    else:
        inputs = torch.rand(mini_batch_size,3,224,224).type(torch.float16).to(device)

    args = parser.parse_args()
    max_flops = args.max_flops
    max_params = args.max_params
    
    if args.algo == "ea_search":
        population_size = args.initial_pop
        num_generations = args.generation_ea
        best_models = search_evolution(population_size=population_size,
                                       num_max_blocks=3,
                                       max_step=num_generations,
                                       metrics=['synflow', 'naswot'],
                                       inputs=inputs,
                                       device=device,
                                       max_flops=max_flops,
                                       max_params=max_params)
        model = best_models[0].get_model()
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
        model = best_models[0].get_model()

    elif args.algo == "our_cnn":
        model = ResNet()

    print("best models ea")
    if len(best_models) > 0:
        for nn in best_models:
            print(f"model: {nn.get_model()}")
            print(f"info flops and params {nn.get_cost_info()}")
            print(f"synflow score: {nn.get_metric_score('synflow')}")
            print(f"naswot score: {nn.get_metric_score('naswot')}")


    

    if args.save:    
            torch.save(model, 'model.pth')

    
    return 






if __name__ == '__main__':
    main()