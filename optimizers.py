import torch

class BasicOptimizer(torch.optim.Optimizer):
  """
  Simple optimizer class based on the SGD optimizer.
  """
  def __init__(self, params, lr:float = 0.01, weight_decay:float = 0):
    """
    Initializes a basic optimizer object.

    Arguments:
    - params (generator): Generator for torch model parameters.
    - lr (float, optional): Learning rate.
    - weight_decay (float, optional): Weight decay.
    """

    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")

    defaults = dict(
        lr=lr,
        weight_decay=weight_decay,
        )

    super().__init__(params, defaults)

  def step(self):
      """
      Performs a single optimization step.
      """

      for group in self.param_groups:
        for p in group["params"]:

          # only update parameters with gradients
          if p.grad is not None:

            # apply weight decay to gradient, if applicable
            if group["weight_decay"] != 0:
              p.grad = p.grad.add(p, alpha=group["weight_decay"])

            # apply gradient-based update
            p.data.add_(p.grad, alpha=-group["lr"])


class KolenPollackOptimizer(BasicOptimizer):
    def __init__(self, params, lr: float = 0.01, weight_decay: float = 0):
        super().__init__(params, lr, weight_decay)
        self.param_dict = {name: param for name, param in params}

    def step(self):
        super().step()  # Perform the standard optimization step

        # Symmetric update for backward weights
        for name, param in self.param_dict.items():
            if 'backward' in name:
                forward_name = name.replace('backward_', '')
                forward_param = self.param_dict.get(forward_name)
                if forward_param is not None:
                    param.data.copy_(forward_param.data)