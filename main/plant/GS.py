import torch
from torch import Tensor

from botorch.test_functions import Levy
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

import gpytorch

class SparseGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.shape[1])
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Power():
    def __init__(self):
        self.surrogate = torch.load('surrogate.pth', weights_only=False,map_location=device).to(device)
        self.surrogate.eval()
        self.NOISE_SE = 0.23
        self.dim = 4
    
    def evaluate_true(self, X):
        return self.surrogate(X).mean.data
    
    def observe(self, X):
        truth = self.evaluate_true(X)
        return truth + torch.randn_like(truth) * self.NOISE_SE

def compute_mean_and_sigma(model, X, compute_mean: bool = True, compute_sigma: bool = True, min_var: float = 1e-12):
    mean, sigma = None, None
    posterior = model.posterior(X=X)

    if compute_mean:
        mean = posterior.mean.squeeze(-2).squeeze(-1)
    if compute_sigma:
        sigma = posterior.variance.clamp_min(min_var).sqrt().view(mean.shape)

    return mean, sigma


class GS(AnalyticAcquisitionFunction):

    def __init__(
        self,
        model,
        **kwargs,
    ) -> None:

        super().__init__(model=model, **kwargs)
        self.beta = torch.tensor(beta)
        self.current_loc = self.model.train_inputs[0][...,-1:,:]

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = compute_mean_and_sigma(self.model, X)
        dist = (unnormalize(X, bounds=bounds) - unnormalize(self.current_loc, bounds=bounds)).norm(dim=-1).squeeze() / Normalize_y
        return mean + self.beta.sqrt() * sigma.squeeze() - dist

class agent():
    
    def __init__(self):
        self.local_X = torch.tensor([]).double().to(device)
        self.local_Y = torch.tensor([]).double().to(device)
        self.reward_record = []
        self.travel_record = []

    def generate_initial_data(self, n = 1):
        self.next_samples =  torch.rand(n,dim).double().to(device)
        new_y = blackbox.observe(self.next_samples)/Normalize_y
        self.local_X = torch.cat([self.local_X, self.next_samples])
        self.local_Y = torch.cat([self.local_Y, new_y])
        

        reward = blackbox.evaluate_true(self.next_samples).mean().item()
        self.next_samples = torch.tensor([]).double().to(device) # clear the batch buffer

        self.update_model() # update model

        print("\rSamples Taken: {} | avg reward: {:.2f}"
              .format(self.local_X.shape[0], reward)
              )  


    def go(self):
        self.plan_route()
        # observe the response of the current batch
        new_y = blackbox.observe(self.next_samples)/Normalize_y
        reward = blackbox.evaluate_true(self.next_samples)
        self.reward_record += reward.cpu().tolist()
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
        diffs = destinations_real[1:]  - destinations_real[:-1]
        self.travel_record += torch.norm(diffs, dim=1).cpu().tolist()

    def plan_next_batch(self):
        acq_func = GS(self.model)
                
        # optimize acquisition function
        candidates, value = optimize_acqf(
            acq_function = acq_func,
            bounds = unit_bounds,
            q = 1,
            num_restarts = NUM_RESTARTS,
            raw_samples = RAW_SAMPLES,  # used for intialization heuristic
            options = {"batch_limit": BATCH_LIMIT, "maxiter": MAX_ITR},
        )
        
        # observe new values 
        self.next_samples = candidates


    
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
        
        self.BOD = candidates

        return blackbox.evaluate_true(self.BOD).item()
    
    
    def report(self,):
        report_len = len(self.reward_record)
        if report_len % 10 == 0:
            current_avg_travel = sum(self.travel_record[-report_len:]) / report_len
            current_avg_reg = sum(self.reward_record[-report_len:])  / report_len
            print("\rSamples Taken: {} | avg travel cost: {:.2f} | avg reward: {:.2f} | BOD reward: {:.2f}"
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
    blackbox = Power()
    dim = blackbox.dim
    # Temperature, Vacuum, Pressure, Humidity 
    bounds = torch.tensor([[1.81,25.36,992.89,25.56],
                           [37.11,81.56,1033.30,100.16]]).double().to(device)
    
    unit_bounds = torch.tensor([[0] * dim, [1] * dim]).double().to(device)
    initial_samples = 2*dim

    # Contorlling parameters (affects peformance)
    beta = 4 #.sqrt()

    T = 200
    cum_reward_table = []
    cum_travel_table = []

    for rep in range(10):
        seed = 600 + rep
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
        cum_reward_table.append(traveler.reward_record)
        cum_travel_table.append(traveler.travel_record)

    import pandas as pd
    pd.DataFrame(cum_reward_table).T.to_excel("GS_reg.xlsx", index=False, engine='openpyxl')

    pd.DataFrame(cum_travel_table).T.to_excel("GS_travel.xlsx", index=False, engine='openpyxl')

