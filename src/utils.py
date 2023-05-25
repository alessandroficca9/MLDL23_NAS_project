
from metrics.metrics import *
import torch.nn as nn 


def get_ranking_based_on_metrics(networks, input,device='cpu'):

    synflows_scores = [ (model, compute_synflow_per_weight(model,input,device)) for model in networks]
    naswot_scores = [ (model, compute_naswot_score(model, input, device) ) for model in networks]

    networks_synflow_scores_sorted = sorted(synflows_scores,key=lambda x: x[1],reverse=True)
    networks_naswot_scores_sorted = sorted(naswot_scores, key=lambda x: x[1], reverse=True)


    #print("synflow")
    #print(networks_synflow_scores_sorted)
    #print("naswot")
    #print(networks_naswot_scores_sorted)

    networks_total_rank = []
    for i,model_nas in enumerate(networks_naswot_scores_sorted):
        for j, model_syn in enumerate(networks_synflow_scores_sorted):
            if model_nas[0] == model_syn[0]:
                networks_total_rank.append( (model_nas[0], i+j) )

    networks_total_rank_sorted = sorted(networks_total_rank, key=lambda x: x[1])

    #print(networks_total_rank_sorted)
    #print(len(networks_total_rank_sorted))
    return [ model[0] for model in networks_total_rank_sorted]

def get_top_k_models(networks, k):
    return networks[:k]

def isfeasible(model: nn.Module,input: torch.Tensor, max_flops: int, max_params: int):
    
    params = get_params(model)
    flops = count_flops(model, input)

    if params <= max_params and flops <= max_flops:
        return True 
    return False


