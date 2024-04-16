
import torch
import numpy as np

class RMSNorm(torch.nn.Module):
    def __init__(self, num_features, epsilon=1e-8, weight=None):
        super(RMSNorm, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        if weight is None:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.weight.data.fill_(1)
   
    def forward(self, x):
        # x.shape (*, d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1) + self.epsilon)   # (*,)
        result = x / rms.unsqueeze(-1) * self.weight  # (*, d_model)
        return result
    
class GeLU(torch.nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()
        # self.register_buffer('sqrt_two_over_pi', torch.sqrt(torch.tensor(2 / torch.pi)))
        # self.register_buffer('coeff', torch.tensor(0.044715))

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
        return x * cdf

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, activation=GeLU()):
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = torch.nn.Linear(d_ff, d_model, bias=False)
        self.activation = activation 
    
    def forward(self, x):
        # x.shape (*, d_model)
        result = self.activation(self.linear1(x))
        result = self.linear2(result)
        return result
    