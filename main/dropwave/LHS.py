import torch
from torch import Tensor

from botorch.test_functions import DropWave
from botorch.optim import optimize_acqf
from botorch.acquisition import PosteriorMean, AnalyticAcquisitionFunction
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import t_batch_mode_transform, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import random

import warnings
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import combinations
from collections import defaultdict
def lhs(n: int, d: int):
    cut = np.linspace(0, 1, n + 1)

    u = np.random.rand(n, d)
    a = cut[:n]
    b = cut[1:]
    samples = a[:, None] + u * (b - a)[:, None]

    for j in range(d):
        np.random.shuffle(samples[:, j])

    return torch.tensor(samples).to(device)

def christofides(distance_matrix, start = 0):
    n = len(distance_matrix)

    if (distance_matrix == 0).all():
        return [i for i in range(n)], 0
    
    mst = minimum_spanning_tree(distance_matrix).toarray()

    degrees = np.count_nonzero(mst, axis=0) + np.count_nonzero(mst, axis=1)
    odd_nodes = [i for i in range(n) if degrees[i] % 2 != 0]

    def find_min_weight_matching(odd_nodes):
        pairs = list(combinations(odd_nodes, 2))
        pairs.sort(key=lambda x: distance_matrix[x[0]][x[1]])
        matching = []
        used = set()
        for pair in pairs:
            if pair[0] not in used and pair[1] not in used:
                matching.append(pair)
                used.add(pair[0])
                used.add(pair[1])
        return matching

    matching = find_min_weight_matching(odd_nodes)

    multigraph = defaultdict(list)
    for i in range(n):
        for j in range(n):
            if mst[i][j] != 0:
                multigraph[i].append(j)
                multigraph[j].append(i)
    for pair in matching:
        multigraph[pair[0]].append(pair[1])
        multigraph[pair[1]].append(pair[0])

    def find_eulerian_circuit(multigraph, start):
        stack = [start]
        path = []
        while stack:
            node = stack[-1]
            if multigraph[node]:
                next_node = multigraph[node].pop()
                stack.append(next_node)
            else:
                path.append(stack.pop())
        return path[::-1]

    eulerian_circuit = find_eulerian_circuit(multigraph, start)

    visited = set()
    hamiltonian_path = []
    for node in eulerian_circuit:
        if node not in visited:
            hamiltonian_path.append(node)
            visited.add(node)

    total_distance = 0
    for i in range(len(hamiltonian_path) - 1):
        total_distance += distance_matrix[hamiltonian_path[i]][hamiltonian_path[i + 1]]

    return hamiltonian_path, total_distance


class agent():
    
    def __init__(self):
        self.local_X = torch.tensor([]).double().to(device)
        self.local_Y = torch.tensor([]).double().to(device)
        self.regret_record = []
        self.travel_record = []

    def generate_initial_data(self, n = 1):
        self.next_samples =  torch.rand(n,dim).double().to(device)
        new_y = blackbox(unnormalize(self.next_samples, bounds))/Normalize_y
        self.local_X = torch.cat([self.local_X, self.next_samples])
        self.local_Y = torch.cat([self.local_Y, new_y])
        

        regret = (blackbox.evaluate_true(unnormalize(self.next_samples, bounds)) - blackbox.evaluate_true(blackbox.optimizers[0]) ).abs().mean().item()
        self.next_samples = torch.tensor([]).double().to(device) # clear the batch buffer

        self.update_model() # update model

        print("\rSamples Taken: {} | avg regret: {:.2f}"
              .format(self.local_X.shape[0], regret)
              )  


    def go(self):
        self.plan_route()
        # observe the response of the current batch
        new_y = blackbox(unnormalize(self.next_samples, bounds))/Normalize_y
        regret = (blackbox.evaluate_true(unnormalize(self.next_samples, bounds)) - blackbox.evaluate_true(blackbox.optimizers[0]) ).abs()
        self.regret_record += regret.cpu().tolist()
        # augment training dataset
        self.local_X = torch.cat([self.local_X, self.next_samples])
        self.local_Y = torch.cat([self.local_Y, new_y])

        self.update_model() # update model

        self.next_samples = torch.tensor([]).double().to(device) # clear the batch buffer


    def update_model(self):      
        targets = self.local_Y.unsqueeze(-1)
        self.model = SingleTaskGP(self.local_X, targets).to(device)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).to(self.local_X)
        fit_gpytorch_mll(self.mll)
    
    def plan_route(self):
        if self.local_X.numel() > 0:
            destinations = torch.cat([self.local_X[-1,:].reshape(1,-1), self.next_samples])
        else:
            destinations = torch.cat([ torch.zeros(1,dim).double().to(device), self.next_samples])
        
        destinations_real = unnormalize(destinations, bounds=bounds)
        distance_matrix = (destinations_real[:, None] - destinations_real[None, :]).norm(dim=-1)
        xopt , travel_cost = christofides(distance_matrix.cpu().numpy())
        diffs = destinations_real[xopt[1:]]  - destinations_real[xopt[:-1]] 
        self.travel_record += torch.norm(diffs, dim=1).cpu().tolist()
        self.next_samples = destinations[xopt[1:]] 

    def plan_next_batch(self):
        self.next_samples = lhs(T,dim)
   
    def eval_BOD(self):
        criteria = PosteriorMean(self.model,)
        candidates, value = optimize_acqf(
            acq_function = criteria,
            bounds = unit_bounds,
            q = 1,
            num_restarts = NUM_RESTARTS,
            raw_samples = RAW_SAMPLES,  # used for intialization heuristic
            options = {"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
        )

        return (blackbox.evaluate_true(unnormalize(candidates, bounds)) - blackbox.evaluate_true(blackbox.optimizers[0]) ).abs().item()
    
    
    def report(self,):
        report_len = len(self.regret_record)
        current_avg_travel = sum(self.travel_record[-report_len:]) / report_len
        current_avg_reg = sum(self.regret_record[-report_len:])  / report_len
        print("\rSamples Taken: {} | avg travel cost: {:.2f} | avg regret: {:.2f} | BOD regret: {:.2f}"
              .format(self.local_X.shape[0] - initial_samples, current_avg_travel, current_avg_reg, self.eval_BOD())
              )       



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    warnings.filterwarnings('ignore', category = BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category = RuntimeWarning)
    warnings.filterwarnings('ignore', category=InputDataWarning)
    
    # Numerical parameters (only affects the precision)
    # for optimizing acquisition functions
    NUM_RESTARTS = 100
    RAW_SAMPLES = 5000
    BATCH_LIMIT = 5000
    MAX_ITR = 50
    
    # for GP
    Normalize_y = 1
    
    # Setting parameters
    NOISE_SE = 0.01
    blackbox = DropWave(negate=True,noise_std = NOISE_SE).to(device)
    dim = blackbox.dim
    bounds = torch.tensor([[-5.21] * dim, [5.21] * dim]).double().to(device)
    
    unit_bounds = torch.tensor([[0] * dim, [1] * dim]).double().to(device)
    initial_samples = 2*dim

    # Contorlling parameters (affects peformance)
    beta = 4 #.sqrt()

    T = 200
    cum_regret_table = []
    cum_travel_table = []

    for rep in range(10):
        seed = 100 + rep
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        traveler = agent()

        traveler.generate_initial_data(n=initial_samples)

        while traveler.local_X.shape[0] - initial_samples < T : 
            traveler.plan_next_batch()
            traveler.go()
            traveler.report()

        print()
        cum_regret_table.append(traveler.regret_record)
        cum_travel_table.append(traveler.travel_record)

    import pandas as pd
    pd.DataFrame(cum_regret_table).T.to_excel("LHS_reg.xlsx", index=False, engine='openpyxl')
    pd.DataFrame(cum_travel_table).T.to_excel("LHS_travel.xlsx", index=False, engine='openpyxl')