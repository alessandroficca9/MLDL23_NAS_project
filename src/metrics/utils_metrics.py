
from metrics.metrics import compute_naswot_score, compute_synflow_per_weight, count_flops, get_params
#from exemplar import Exemplar


def compute_metrics(exemplar, inputs, device):

    if exemplar.metrics == None:
        exemplar.metrics = {}
        exemplar.metrics["synflow"] = compute_synflow_per_weight(net=exemplar.get_model(), inputs=inputs[0], device=device)
        # inputs is batch dataloader -> input[0] list of all tensors without labels
        exemplar.metrics["naswot"] = compute_naswot_score(net=exemplar.get_model(), inputs=inputs[0], device=device)

    if exemplar.cost_info == None:
        exemplar.cost_info = {}
        exemplar.cost_info['FLOPS'] = count_flops(model=exemplar.get_model(), input=inputs[0], device=device)
        exemplar.cost_info["#Parameters"] = get_params(model=exemplar.get_model())

    return 


def compute_metrics_population(population, inputs, device):

    for exemplar in population:
        compute_metrics(exemplar=exemplar, inputs=inputs, device=device)

    return 
    
def isfeasible(exemplar, max_params: int, max_flops: int, inputs, device): 

    if exemplar.get_cost_info() == None:
        compute_metrics(exemplar=exemplar, inputs=inputs, device=device)

    cost_info = exemplar.get_cost_info()

    if cost_info["FLOPS"] <= max_flops and cost_info["#Parameters"] <= max_params:
        return True
    else: 
        return False