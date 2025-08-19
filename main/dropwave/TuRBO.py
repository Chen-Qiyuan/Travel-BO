import torch
import math
from dataclasses import dataclass
from botorch.test_functions import DropWave
from botorch.optim import optimize_acqf
from botorch.acquisition import PosteriorMean, UpperConfidenceBound
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import t_batch_mode_transform, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import random
from contextlib import ExitStack
from torch.quasirandom import SobolEngine
import gpytorch.settings as gpts
from botorch.generation import MaxPosteriorSampling
import warnings
import numpy as np
@dataclass
class TurboState:
    dim: int
    batch_size: int = 1
    length: float = 0.8
    length_max: float = 1.6
    length_min: float = 0.5 ** 7
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(float(self.dim) / self.batch_size)


class agent():
    
    def __init__(self):
        self.local_X = torch.tensor([]).double().to(device)
        self.local_Y = torch.tensor([]).double().to(device)
        self.X = torch.tensor([]).double().to(device)
        self.Y = torch.tensor([]).double().to(device)
        self.regret_record = []
        self.travel_record = []

    def update_state(self, state, Y_next):
        if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
            state.success_counter += 1
            state.failure_counter = 0
        else:
            state.success_counter = 0
            state.failure_counter += 1

        if state.success_counter == state.success_tolerance:  # Expand trust region
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
        elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
            state.length /= 2.0
            state.failure_counter = 0

        state.best_value = max(state.best_value, max(Y_next).item())
        if state.length < state.length_min:
            state.restart_triggered = True
        return state

    def generate_initial_data(self, n = 1):
        self.next_samples =  torch.rand(n,dim).double().to(device)
        new_y = blackbox(unnormalize(self.next_samples, bounds))/Normalize_y
        self.local_X = torch.cat([self.local_X, self.next_samples])
        self.local_Y = torch.cat([self.local_Y, new_y])
        self.X = torch.cat([self.X, self.next_samples])
        self.Y = torch.cat([self.Y, new_y])        
        self.state = TurboState(dim, batch_size=1, best_value=max(self.local_Y).item())

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
        self.X = torch.cat([self.X, self.next_samples])
        self.Y = torch.cat([self.Y, new_y])   
        self.update_model() # update model
        self.state = self.update_state(state=self.state, Y_next=new_y)

        self.next_samples = torch.tensor([]).double().to(device) # clear the batch buffer


    def update_model(self):      
        targets = self.local_Y.unsqueeze(-1)
        self.model = SingleTaskGP(self.local_X, targets).to(device)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model).to(self.local_X)
        fit_gpytorch_mll(self.mll)
    
    def plan_route(self):
        if self.X.numel() > 0:
            destinations = torch.cat([self.X[-1,:].reshape(1,-1), self.next_samples])
        else:
            destinations = torch.cat([ torch.zeros(1,dim).double().to(device), self.next_samples])
        
        destinations_real = unnormalize(destinations, bounds=bounds)
        diffs = destinations_real[1:]  - destinations_real[:-1]
        self.travel_record += torch.norm(diffs, dim=1).cpu().tolist()

    def plan_next_batch(self):
        if self.state.restart_triggered:
            self.next_samples = torch.rand(2*dim,dim).double().to(device)
            self.local_X = torch.tensor([]).double().to(device)
            self.local_Y = torch.tensor([]).double().to(device)
            self.state = TurboState(dim)

        else:
            x_center = self.local_X[self.local_Y.argmax(), :].clone()
            weights = self.model.covar_module.lengthscale.squeeze().detach()
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            tr_lb = torch.clamp(x_center - weights * self.state.length / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(x_center + weights * self.state.length / 2.0, 0.0, 1.0)
            sobol = SobolEngine(self.local_X.shape[-1], scramble=True)
            X_cand = unnormalize(sobol.draw(1000).to(device).double(), torch.stack([tr_lb, tr_ub]))
            # Thompson sample
            with ExitStack() as es:
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
                thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
                candidates = thompson_sampling(X_cand, num_samples=1)          
            
            self.next_samples = torch.cat([self.next_samples, candidates.detach()])


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
        if report_len % 10 == 0:
            current_avg_travel = sum(self.travel_record[-report_len:]) / report_len
            current_avg_reg = sum(self.regret_record[-report_len:])  / report_len
            print("\rSamples Taken: {} | avg travel cost: {:.2f} | avg regret: {:.2f} | BOD regret: {:.2f}"
                .format(self.X.shape[0] - initial_samples, current_avg_travel, current_avg_reg, self.eval_BOD())
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

        while traveler.X.shape[0] - initial_samples < T : 
            traveler.plan_next_batch()
            traveler.go()
            traveler.report()

        print()
        cum_regret_table.append(traveler.regret_record)
        cum_travel_table.append(traveler.travel_record)

    import pandas as pd
    pd.DataFrame(cum_regret_table).T.to_excel("TuRBO_reg.xlsx", index=False, engine='openpyxl')
    pd.DataFrame(cum_travel_table).T.to_excel("TuRBO_travel.xlsx", index=False, engine='openpyxl')