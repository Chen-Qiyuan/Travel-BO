import torch
from torch import Tensor

from botorch.test_functions import Hartmann, Cosine8, Levy, Ackley, Branin, DropWave, EggHolder, Shekel, Powell, Griewank
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.acquisition import PosteriorMean, AnalyticAcquisitionFunction, UpperConfidenceBound, AcquisitionFunction
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
from botorch.models.gp_regression import FixedNoiseGP
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
from python_tsp.heuristics import solve_tsp_local_search
from contextlib import ExitStack
from torch.quasirandom import SobolEngine
import gpytorch.settings as gpts
from botorch.generation import MaxPosteriorSampling
import matplotlib.pyplot as plt

def compute_mean_and_sigma(model, X, compute_mean: bool = True, compute_sigma: bool = True, min_var: float = 1e-12):
    mean, sigma = None, None
    posterior = model.posterior(X=X)

    if compute_mean:
        mean = posterior.mean.squeeze(-2).squeeze(-1)
    if compute_sigma:
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)

    return mean, sigma

class agent():
    
    def __init__(self):
        # keep current model
        self.model = None
        self.value = []
        self.total_size = 0

        # keep history of models to keep track of feasible regions.
        self.model_history = []
        self.LCB_history = []

        # init training data 
        self.local_X = torch.tensor([]).double().cuda()
        self.local_Y = torch.tensor([]).double().cuda()

        self.cumulative_traveling_cost = 0
        self.cumulative_regret = 0
        self.previous_traveling_cost = 0
        self.previous_cumulative_regret = 0
        self.current_batch_size = initial_batch_size
        self.cum_regret_record = []
        self.cum_travel_record = []


    def generate_initial_data(self, n = 1):
        self.next_samples =  torch.zeros(1,dim).double().cuda()
        self.go()


    def BOD_reward(self):
        return blackbox.evaluate_true(unnormalize(self.get_BOD(), bounds))/Normalize_y
    
    def go(self):
        self.plan_route()
        # observe the response of the current batch
        new_y = blackbox(unnormalize(self.next_samples, bounds))/Normalize_y
        regret = (blackbox.evaluate_true(unnormalize(self.next_samples, bounds)) + blackbox.optimal_value * negate ).abs().sum()
        self.cumulative_regret += regret
        # augment training dataset
        self.local_X = torch.cat([self.local_X, self.next_samples])
        self.local_Y = torch.cat([self.local_Y, new_y])
        self.update_model() # update model

        self.next_samples = torch.tensor([]).double().cuda() # clear the batch buffer

    def update_model(self):      
        nug = NOISE_SE/Normalize_y
        self.model = FixedNoiseGP(self.local_X, self.local_Y.unsqueeze(-1), torch.ones_like(self.local_Y.unsqueeze(-1)) * nug).cuda()

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).cuda()
        fit_gpytorch_mll(self.mll)
    
    def plan_route(self):
        if self.local_X.numel() > 0:
            destinations = torch.cat([self.local_X[-1,:].reshape(1,-1), self.next_samples])
        else:
            destinations = torch.cat([ torch.zeros(1,dim).double().cuda(), self.next_samples])
        
        destinations_real = unnormalize(destinations, bounds=bounds)

        distance_matrix = (destinations_real[:, None] - destinations_real[None, :]).norm(dim=-1)
        xopt, fopt = solve_tsp_local_search(distance_matrix.cpu().numpy(), 0)
        self.cumulative_traveling_cost += fopt
        self.next_samples = destinations[xopt[1:]] 

    def plan_next_batch(self):
        sobol = SobolEngine(self.local_X.shape[-1], scramble=True)
        X_cand = sobol.draw(1000).cuda().double()
        # Thompson sample
        with ExitStack() as es:
            es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            candidates = thompson_sampling(X_cand, num_samples=1)          
        
        self.next_samples = torch.cat([self.next_samples, candidates.detach()])
    
    def get_BOD(self):
        criteria = PosteriorMean(self.model)
        candidates, value = optimize_acqf(
            acq_function = criteria,
            bounds = unit_bounds,
            q = 1,
            num_restarts = NUM_RESTARTS,
            raw_samples = RAW_SAMPLES,  # used for intialization heuristic
            options = {"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
        )

        self.BOD = candidates
        return candidates
    
    
    def report(self,):
        if self.local_X.shape[0] == self.total_size + self.current_batch_size or self.local_X.shape[0] == T:
            self.total_size = self.local_X.shape[0]
            self.previous_batch_size = self.current_batch_size + 0
            self.current_batch_size = self.previous_batch_size * increment # set next batch size
        else:
            print("\rSamples Taken: "+str(self.local_X.shape[0]),end="")
            return
        # Separate the list of points into two lists of x and y coordinates
        current_avg_travel = (self.cumulative_traveling_cost - self.previous_traveling_cost) / self.previous_batch_size
        current_avg_reg = (self.cumulative_regret.item() - self.previous_cumulative_regret) / self.previous_batch_size
        print("\rSamples Taken: {} | avg travel cost: {:.2f} | avg regret: {:.2f}"
              .format(self.local_X.shape[0], current_avg_travel, current_avg_reg)
              )
        
        self.previous_traveling_cost = self.cumulative_traveling_cost + 0
        self.previous_cumulative_regret = self.cumulative_regret.item() + 0
        
        self.cum_regret_record.append(current_avg_travel)
        self.cum_travel_record.append(current_avg_reg)



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    warnings.filterwarnings('ignore', category = BadInitialCandidatesWarning)
    warnings.filterwarnings('ignore', category = RuntimeWarning)
    warnings.filterwarnings('ignore', category=InputDataWarning)
    
    # Numerical parameters (only affects the precision)
    # for optimizing acquisition functions
    NUM_RESTARTS = 500
    RAW_SAMPLES = 5000
    BATCH_LIMIT = 500
    MAX_ITR = 10
    # for GP
    Normalize_y = 1
    
    # Setting parameters
    NOISE_SE = 0.1
    negate = -1
    blackbox = Griewank(negate=True,noise_std = NOISE_SE)
    dim = blackbox.dim
    # bounds = torch.tensor(blackbox._bounds).double().cuda().T
    bounds = torch.tensor([[-10]*dim,[10]*dim]).double().cuda()

    unit_bounds = torch.tensor([[0] * dim, [1] * dim]).double().cuda()
    initial_samples = 1
    initial_batch_size = 1
    increment = 2
    cost_mult = 1


    T = 511
    cum_regret_table = []
    cum_travel_table = []

    for rep in range(5):
        seed = 300 + rep
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        traveler = agent()

        traveler.generate_initial_data(n=initial_samples)
        BOD_reward = traveler.report()

        while traveler.local_X.shape[0] < T: 
            traveler.plan_next_batch()
            traveler.go()
            traveler.report()

        print()
        cum_regret_table.append(traveler.cum_regret_record)
        cum_travel_table.append(traveler.cum_travel_record)

    import pandas as pd
    pd.DataFrame(cum_regret_table).T.to_excel("TS_reg.xlsx", index=False, engine='openpyxl')
    pd.DataFrame(cum_travel_table).T.to_excel("TS_travel.xlsx", index=False, engine='openpyxl')