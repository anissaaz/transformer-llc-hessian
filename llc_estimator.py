import torch
import numpy as np

class LLCEstimator:
    """
    Calculates LLC using the formula: lambda = n * beta * (E[L] - L*)
    """
    def __init__(self, num_samples, beta, init_loss):
        self.num_samples = num_samples
        self.beta = beta
        self.init_loss = init_loss
        self.loss_trace = []

    def update(self, loss):
        self.loss_trace.append(loss)
    
    def estimate(self):
        if not self.loss_trace:
            return {}

        avg_loss = np.mean(self.loss_trace)     # approximation for E[L]
        
        #LLC Calculation: lambda_hat = n * beta * (mean_loss - init_loss)
        llc = self.num_samples * self.beta * (avg_loss - self.init_loss)
        
        return {
            "llc": llc,
            "loss_avg": avg_loss,
            "loss_init": self.init_loss
        }
    