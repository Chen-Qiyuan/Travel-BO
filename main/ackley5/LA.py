import torch
from torch import Tensor

from botorch.test_functions import Ackley
from botorch.optim import optimize_acqf
from botorch.acquisition import PosteriorMean, AnalyticAcquisitionFunction
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import t_batch_mode_transform, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import random
from botorch.acquisition import qMultiStepLookahead
from botorch.acquisition.analytic import _scaled_improvement, _ei_helper
from botorch.acquisition.multi_step_lookahead import make_best_f, warmstart_multistep

import warnings
import numpy as np

class WEI(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model,
        best_f,
        posterior_transform = None,
        maximize: bool = True,
    ):

        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.maximize = maximize
        self.current_loc = self.model.train_inputs[0][...,-1:,:]


    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        dist = (unnormalize(X, bounds=bounds) - unnormalize(self.current_loc, bounds=bounds)).norm(dim=-1).squeeze() / Normalize_y
        return sigma * _ei_helper(u) - dist
    
class agent():
    
    def __init__(self):
        self.model_history = []
        self.LCB_history = []
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
        diffs = destinations_real[1:]  - destinations_real[:-1]
        self.travel_record += torch.norm(diffs, dim=1).cpu().tolist()

    def plan_next_batch(self):
        q = 1
        q_batch_sizes = [1, 1, 1]
        num_fantasies = [5, 5, 1]

        qMS = qMultiStepLookahead(
            model=self.model,
            batch_sizes=q_batch_sizes,
            valfunc_cls=[WEI] * (len(q_batch_sizes)+1),
            valfunc_argfacs=[make_best_f] * (len(q_batch_sizes)+1),
            num_fantasies=num_fantasies,
        )

        candidates, value = optimize_acqf(
            acq_function = qMS,
            bounds = unit_bounds,
            q = qMS.get_augmented_q_batch_size(q),
            num_restarts = 5,
            raw_samples = 20,  # used for intialization heuristic
            options = {"batch_limit": 5, "maxiter": MAX_ITR},
        )

        self.next_samples = candidates
        # print(self.next_samples)

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
    NOISE_SE = 0.1
    blackbox = Ackley(negate=True,noise_std = NOISE_SE, dim = 8).to(device)
    dim = blackbox.dim
    bounds = torch.tensor([[-10] * dim, [10] * dim]).double().to(device)
    
    unit_bounds = torch.tensor([[0] * dim, [1] * dim]).double().to(device)
    initial_samples = 2*dim

    # Contorlling parameters (affects peformance)
    beta = 4 #.sqrt()

    T = 200
    cum_regret_table = []
    cum_travel_table = []

    for rep in range(10):
        seed = 300 + rep
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
    pd.DataFrame(cum_regret_table).T.to_excel("LA_reg.xlsx", index=False, engine='openpyxl')
    pd.DataFrame(cum_travel_table).T.to_excel("LA_travel.xlsx", index=False, engine='openpyxl')