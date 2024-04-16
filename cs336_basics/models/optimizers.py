import torch 
from typing import Optional, Callable
import math

class SGD(torch.optim.Optimizer):
	def __init__(self, params, lr=1e-3):
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")
		defaults = {"lr": lr}
		super().__init__(params, defaults)
  
	def step(self, closure: Optional[Callable] = None):
		loss = None if closure is None else closure()
		for group in self.param_groups:
			lr = group["lr"] # Get the learning rate.
			for p in group["params"]:
				if p.grad is None:
					continue
				state = self.state[p] # Get state associated with p.
				t = state.get("t", 0) # Get iteration number from the state, or initial value.
				grad = p.grad.data # Get the gradient of loss with respect to p.
				p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
				state["t"] = t + 1 # Increment iteration number.
		return loss


class AdamW(torch.optim.Optimizer):
	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
		if eps < 0:
			raise ValueError(f"Invalid epsilon value: {eps}")
		if weight_decay < 0:
			raise ValueError(f"Invalid weight decay value: {weight_decay}")
		defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
		super().__init__(params, defaults)
  
	def step(self, closure: Optional[Callable] = None):
		loss = None if closure is None else closure()
		for group in self.param_groups:
			lr = group["lr"]
			betas = group["betas"]
			eps = group["eps"]
			weight_decay = group["weight_decay"]
   
			for p in group["params"]:
				if p.grad is None:
					continue
				state = self.state[p]
				grad = p.grad.data
				t = state.get("t", 0)
				m = state.get("m", 0)
				v = state.get("v", 0)
				beta1, beta2 = betas
				m = beta1 * m + (1 - beta1) * grad
				v = beta2 * v + (1 - beta2) * grad**2
				lr_t = lr * math.sqrt(1 - beta2**(t + 1)) / (1 - beta1**(t + 1))
				p.data -= lr_t * m / (torch.sqrt(v) + eps)
				if weight_decay > 0:
					p.data -= lr * weight_decay * p.data
				state["t"] = t + 1
				state["m"] = m
				state["v"] = v
    
		return loss

