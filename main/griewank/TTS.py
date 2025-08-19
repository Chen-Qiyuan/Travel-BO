import torch
from torch import Tensor
from botorch.test_functions import Griewank
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.acquisition import AnalyticAcquisitionFunction, UpperConfidenceBound, AcquisitionFunction
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
from botorch.sampling.normal import SobolQMCNormalSampler, IIDNormalSampler
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import t_batch_mode_transform, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import random
from gpytorch.kernels import ScaleKernel, RBFKernel

from matplotlib.colors import LogNorm, PowerNorm

import warnings
import numpy as np
from copy import deepcopy

from contextlib import ExitStack
from torch.quasirandom import SobolEngine
import gpytorch.settings as gpts
from botorch.generation import MaxPosteriorSampling
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import combinations
from collections import defaultdict

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

def compute_mean_and_sigma(model, X, compute_mean: bool = True, compute_sigma: bool = True, min_var: float = 1e-12):
    mean, sigma = None, None
    posterior = model.posterior(X=X)

    if compute_mean:
        mean = posterior.mean.squeeze(-2).squeeze(-1)
    if compute_sigma:
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)

    return mean, sigma

def feasiblity_ind(X, model_hty, LCB_hty, scale = 1000, max_count = 1):
    indicator = torch.ones(X.shape[0]).to(device)
    n_constraints = min(max_count, len(model_hty)) # alleviate computation if needed
    for count in range(1,n_constraints+1):
        mean, sigma = compute_mean_and_sigma(model_hty[-count], X)
        ucb = mean + torch.tensor(beta).sqrt() * sigma /2
        lcb = LCB_hty[-count]
        indicator *= torch.nn.functional.sigmoid((ucb-lcb)*scale)
    return indicator


def get_feasible_points(N, model_hty, LCB_hty, bounds = None):
    if bounds == None:
        local_bounds = unit_bounds
    X = torch.tensor([]).to(device)
    for attemp in range(10):
        if X.shape[0] == N:
            break
        Xraw = local_bounds[0] + (local_bounds[1] - local_bounds[0]) * torch.rand(10 * N, dim).to(device)
        X_pos = Xraw[(feasiblity_ind(Xraw, model_hty, LCB_hty) > 0.5 )][:N]
        X = torch.cat([X, X_pos])

    if X.shape[0] < 1:
        X = torch.cat([X, traveler.BOD])

    return X.data

class ConstrainedLowerConfidenceBound(AnalyticAcquisitionFunction):

    def __init__(
        self,
        model,
        beta,
        model_history,
        LCB_history,
        **kwargs,
    ) -> None:

        super().__init__(model=model, **kwargs)
        self.model_hty = model_history
        self.LCB_hty = LCB_history
        self.beta = torch.tensor(beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = compute_mean_and_sigma(self.model, X)
        feasiblity = feasiblity_ind(X, self.model_hty, self.LCB_hty, scale=100)
        return (mean - self.beta.sqrt() * sigma) + feasiblity * 10

class PosteriorMean(AnalyticAcquisitionFunction):

    def __init__(
        self,
        model,
        model_history,
        LCB_history,
        maximize = True,
        **kwargs,
    ) -> None:

        super().__init__(model=model, **kwargs)
        self.model_hty = model_history
        self.LCB_hty = LCB_history
        self.beta = torch.tensor(beta)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, _ = compute_mean_and_sigma(self.model, X, compute_sigma = False)
        feasiblity = feasiblity_ind(X, self.model_hty, self.LCB_hty, scale=100)
        return (mean if self.maximize else -mean) + feasiblity * 10


class ConstrainedLowerConfidenceBound(AnalyticAcquisitionFunction):

    def __init__(
        self,
        model,
        beta,
        model_hty,
        LCB_hty,
        **kwargs,
    ) -> None:

        super().__init__(model=model, **kwargs)
        self.model_hty = model_hty
        self.LCB_hty = LCB_hty
        self.beta = torch.tensor(beta)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = compute_mean_and_sigma(self.model, X)
        feasiblity = feasiblity_ind(X, self.model_hty, self.LCB_hty)
        return (mean - self.beta.sqrt() * sigma) + feasiblity * 10


class agent():
    
    def __init__(self):
        self.model_hty = []
        self.LCB_hty = []
        self.local_X = torch.tensor([]).double().to(device)
        self.local_Y = torch.tensor([]).double().to(device)
        self.regret_record = []
        self.travel_record = []
        self.current_batch_size = initial_batch_size


    def generate_initial_data(self, n = 1):
        self.next_samples =  torch.rand(n,dim).double().to(device)
        new_y = blackbox(unnormalize(self.next_samples, bounds))/Normalize_y
        self.local_X = torch.cat([self.local_X, self.next_samples])
        self.local_Y = torch.cat([self.local_Y, new_y])
        regret = (blackbox.evaluate_true(unnormalize(self.next_samples, bounds)) - blackbox.evaluate_true(blackbox.optimizers[0]) ).abs().mean().item()
        self.next_samples = torch.tensor([]).double().to(device) # clear the batch buffer
        self.update_model()

        print("\rSamples Taken: {} | avg regret: {:.2f}"
              .format(self.local_X.shape[0], regret)
              )  

        self.eval_BOD()

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
        self.previous_batch_size = self.current_batch_size
        self.current_batch_size = int(np.ceil(self.previous_batch_size * batch_size_multiplier)) # set next batch size
    
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
        self.get_search_region()
        
        sobol = SobolEngine(self.local_X.shape[-1], scramble=True)
        for _ in range(int(self.current_batch_size)):
            X_cand = torch.tensor([])
            X_cand = sobol.draw(1000).to(device).double()
            feasiblity = feasiblity_ind(X_cand, self.model_hty, self.LCB_hty, scale=1)
            X_cand = X_cand[feasiblity>0.5]
            if X_cand.numel() < 10:
                feas_points = get_feasible_points(NUM_RESTARTS, self.model_hty, self.LCB_hty, bounds = None)
                X_cand = torch.cat([X_cand, feas_points.detach()])

            # Thompson sample
            with ExitStack() as es:
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
                thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
                candidates = thompson_sampling(X_cand, num_samples=1)          
            
            ind = feasiblity_ind(candidates, self.model_hty, self.LCB_hty, scale = 1).item()
            if ind < 0.5:
                print("Warning: Feasibility Issue!")
            self.next_samples = torch.cat([self.next_samples, candidates.detach()])
            print("\rSamples Planed: " + str(self.next_samples.shape[0]), end="")
            if self.local_X.shape[0] + self.next_samples.shape[0] - initial_samples >= T:
                self.current_batch_size = self.next_samples.shape[0]
                break
    
    def eval_BOD(self):
        feas_points = get_feasible_points(NUM_RESTARTS, self.model_hty, self.LCB_hty, bounds = None)
        criteria = PosteriorMean(self.model,self.model_hty,self.LCB_hty)
        candidate, value = optimize_acqf(
            acq_function = criteria,
            bounds = unit_bounds,
            q = 1,
            num_restarts = NUM_RESTARTS,
            batch_initial_conditions = feas_points.unsqueeze(1),
            options = {"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
        )

        self.BOD = candidate

        return (blackbox.evaluate_true(unnormalize(candidate, bounds)) - blackbox.evaluate_true(blackbox.optimizers[0]) ).abs().item()
    

    def get_search_region(self):
        feas_points = get_feasible_points(NUM_RESTARTS, self.model_hty, self.LCB_hty, bounds = None)
        criteria = ConstrainedLowerConfidenceBound(            
            self.model,
            beta,
            self.model_hty, 
            self.LCB_hty,)
        
        candidates, value = optimize_acqf(
            acq_function=criteria,
            bounds=unit_bounds,
            q=1,
            num_restarts=NUM_RESTARTS,
            batch_initial_conditions = feas_points.unsqueeze(1),
            options={"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
        )

        mean, sigma = compute_mean_and_sigma(self.model, candidates)
        self.model_hty.append(deepcopy(self.model))
        self.LCB_hty.append((mean - torch.tensor(beta).sqrt().to(device)*sigma/2).item())

    
    def report(self,):
        report_len = len(self.regret_record)
        current_avg_travel = sum(self.travel_record[-report_len:]) / report_len
        current_avg_reg = sum(self.regret_record[-report_len:])  / report_len
        print("\rSamples Taken: {} | avg travel cost: {:.2f} | avg regret: {:.2f} | BOD regret: {:.2f}"
              .format(self.local_X.shape[0] - initial_samples, current_avg_travel, current_avg_reg, self.eval_BOD())
              )          
    


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dtype = torch.double
    warnings.filterwarnings('ignore', category = BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category = RuntimeWarning)
    warnings.filterwarnings('ignore', category=InputDataWarning)
    
    # only affects the precision, decrease if out of memeory
    NUM_RESTARTS = 100
    RAW_SAMPLES = 5000
    BATCH_LIMIT = 5000
    MAX_ITR = 50

    # for GP
    Normalize_y = 1
    
    # Setting parameters
    NOISE_SE = 0.01
    blackbox = Griewank(negate=True,noise_std = NOISE_SE).to(device)
    dim = blackbox.dim
    bounds = torch.tensor([[-20] * dim, [20] * dim]).double().to(device)

    unit_bounds = torch.tensor([[0] * dim, [1] * dim]).double().to(device)
    initial_samples = 2 * dim # initial number of samples
    initial_batch_size = 1 # starting batch size for batched algorithm
    batch_size_multiplier = 1.1 # next batch size = multiplier * previous batch size 

    # Contorlling parameters (affects peformance)
    beta = 4 #.sqrt()

    T = 200
    cum_regret_table = []
    cum_travel_table = []
    
    for rep in range(10):
        seed = 200 + rep
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
    pd.DataFrame(cum_regret_table).T.to_excel("TTS_reg.xlsx", index=False, engine='openpyxl')
    pd.DataFrame(cum_travel_table).T.to_excel("TTS_travel.xlsx", index=False, engine='openpyxl')