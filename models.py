import numpy as np
import torch
from torch import Tensor
from typing import Optional, Callable, Tuple

class MultiLayerPerceptron(torch.nn.Module):
  """
  Simple multilayer perceptron model class with one hidden layer.
  """

  def __init__(
      self,
      num_inputs,
      num_hidden,
      num_outputs,
      activation_type="sigmoid",
      bias=False,
      ):
    """
    Initializes a multilayer perceptron with a single hidden layer.

    Arguments:
    - num_inputs (int, optional): number of input units (i.e., image size)
    - num_hidden (int, optional): number of hidden units in the hidden layer
    - num_outputs (int, optional): number of output units (i.e., number of
      classes)
    - activation_type (str, optional): type of activation to use for the hidden
      layer ('sigmoid', 'tanh', 'relu' or 'linear')
    - bias (bool, optional): if True, each linear layer will have biases in
      addition to weights
    """


    super().__init__()

    self.num_inputs = num_inputs
    self.num_hidden = num_hidden
    self.num_outputs = num_outputs
    self.activation_type = activation_type
    self.bias = bias

    # default weights (and biases, if applicable) initialization is used
    # see https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
    self.lin1 = torch.nn.Linear(num_inputs, num_hidden, bias=bias)
    self.lin2 = torch.nn.Linear(num_hidden, num_outputs, bias=bias)

    self._store_initial_weights_biases()

    self._set_activation() # activation on the hidden layer
    self.softmax = torch.nn.Softmax(dim=1) # activation on the output layer


  def _store_initial_weights_biases(self):
    """
    Stores a copy of the network's initial weights and biases.
    """

    self.init_lin1_weight = self.lin1.weight.data.clone()
    self.init_lin2_weight = self.lin2.weight.data.clone()
    if self.bias:
      self.init_lin1_bias = self.lin1.bias.data.clone()
      self.init_lin2_bias = self.lin2.bias.data.clone()

  def _set_activation(self):
    """
    Sets the activation function used for the hidden layer.
    """

    if self.activation_type.lower() == "sigmoid":
      self.activation = torch.nn.Sigmoid() # maps to [0, 1]
    elif self.activation_type.lower() == "tanh":
      self.activation = torch.nn.Tanh() # maps to [-1, 1]
    elif self.activation_type.lower() == "relu":
      self.activation = torch.nn.ReLU() # maps to positive
    elif self.activation_type.lower() == "identity":
      self.activation = torch.nn.Identity() # maps to same
    else:
      raise NotImplementedError(
          f"{self.activation_type} activation type not recognized. Only "
          "'sigmoid', 'relu' and 'identity' have been implemented so far."
          )

  def forward(self, X, y=None):
    """
    Runs a forward pass through the network.

    Arguments:
    - X (torch.Tensor): Batch of input images.
    - y (torch.Tensor, optional): Batch of targets. This variable is not used
      here. However, it may be needed for other learning rules, to it is
      included as an argument here for compatibility.

    Returns:
    - y_pred (torch.Tensor): Predicted targets.
    """

    h = self.activation(self.lin1(X.reshape(-1, self.num_inputs)))
    y_pred = self.softmax(self.lin2(h))
    return y_pred

  def forward_backprop(self, X):
    """
    Identical to forward(). Should not be overwritten when creating new
    child classes to implement other learning rules, as this method is used
    to compare the gradients calculated with other learning rules to those
    calculated with backprop.
    """

    h = self.activation(self.lin1(X.reshape(-1, self.num_inputs)))
    y_pred = self.softmax(self.lin2(h))
    return y_pred


  def list_parameters(self):
    """
    Returns a list of model names for a gradient dictionary.

    Returns:
    - params_list (list): List of parameter names.
    """

    params_list = list()

    for layer_str in ["lin1", "lin2"]:
      params_list.append(f"{layer_str}_weight")
      if self.bias:
        params_list.append(f"{layer_str}_bias")

    return params_list


  def gather_gradient_dict(self):
    """
    Gathers a gradient dictionary for the model's parameters. Raises a
    runtime error if any parameters have no gradients.

    Returns:
    - gradient_dict (dict): A dictionary of gradients for each parameter.
    """

    params_list = self.list_parameters()

    gradient_dict = dict()
    for param_name in params_list:
      layer_str, param_str = param_name.split("_")
      layer = getattr(self, layer_str)
      grad = getattr(layer, param_str).grad
      if grad is None:
        raise RuntimeError("No gradient was computed")
      gradient_dict[param_name] = grad.detach().clone().numpy()

    return gradient_dict


class HebbianFunction(torch.autograd.Function):
  """
  Gradient computing function class for Hebbian learning.
  """

  @staticmethod
  def forward(context, input, weight, bias=None, nonlinearity=None, target=None):
    """
    Forward pass method for the layer. Computes the output of the layer and
    stores variables needed for the backward pass.

    Arguments:
    - context (torch context): context in which variables can be stored for
      the backward pass.
    - input (torch tensor): input to the layer.
    - weight (torch tensor): layer weights.
    - bias (torch tensor, optional): layer biases.
    - nonlinearity (torch functional, optional): nonlinearity for the layer.
    - target (torch tensor, optional): layer target, if applicable.

    Returns:
    - output (torch tensor): layer output.
    """

    # compute the output for the layer (linear layer with non-linearity)
    output = input.mm(weight.t())
    if bias is not None:
      output += bias.unsqueeze(0).expand_as(output)
    if nonlinearity is not None:
      output = nonlinearity(output)

    # calculate the output to use for the backward pass
    output_for_update = output if target is None else target

    # store variables in the context for the backward pass
    context.save_for_backward(input, weight, bias, output_for_update)

    return output

  @staticmethod
  def backward(context, grad_output=None):
    """
    Backward pass method for the layer. Computes and returns the gradients for
    all variables passed to forward (returning None if not applicable).

    Arguments:
    - context (torch context): context in which variables can be stored for
      the backward pass.
    - input (torch tensor): input to the layer.
    - weight (torch tensor): layer weights.
    - bias (torch tensor, optional): layer biases.
    - nonlinearity (torch functional, optional): nonlinearity for the layer.
    - target (torch tensor, optional): layer target, if applicable.

    Returns:
    - grad_input (None): gradients for the input (None, since gradients are not
      backpropagated in Hebbian learning).
    - grad_weight (torch tensor): gradients for the weights.
    - grad_bias (torch tensor or None): gradients for the biases, if they aren't
      None.
    - grad_nonlinearity (None): gradients for the nonlinearity (None, since
      gradients do not apply to the non-linearities).
    - grad_target (None): gradients for the targets (None, since
      gradients do not apply to the targets).
    """

    input, weight, bias, output_for_update = context.saved_tensors
    grad_input = None
    grad_weight = None
    grad_bias = None
    grad_nonlinearity = None
    grad_target = None

    input_needs_grad = context.needs_input_grad[0]
    if input_needs_grad:
      pass

    weight_needs_grad = context.needs_input_grad[1]
    if weight_needs_grad:
      grad_weight = output_for_update.t().mm(input)
      grad_weight = grad_weight / len(input) # average across batch

      # center around 0
      grad_weight = grad_weight - grad_weight.mean(axis=0) # center around 0

      ## or apply Oja's rule (not compatible with clamping outputs to the targets!)
      # oja_subtract = output_for_update.pow(2).mm(grad_weight).mean(axis=0)
      # grad_weight = grad_weight - oja_subtract

      # take the negative, as the gradient will be subtracted
      grad_weight = -grad_weight

    if bias is not None:
      bias_needs_grad = context.needs_input_grad[2]
      if bias_needs_grad:
        grad_bias = output_for_update.mean(axis=0) # average across batch

        # center around 0
        grad_bias = grad_bias - grad_bias.mean()

        ## or apply an adaptation of Oja's rule for biases
        ## (not compatible with clamping outputs to the targets!)
        # oja_subtract = (output_for_update.pow(2) * bias).mean(axis=0)
        # grad_bias = grad_bias - oja_subtract

        # take the negative, as the gradient will be subtracted
        grad_bias = -grad_bias

    return grad_input, grad_weight, grad_bias, grad_nonlinearity, grad_target

class HebbianMultiLayerPerceptron(MultiLayerPerceptron):
  """
  Hebbian multilayer perceptron with one hidden layer.
  """

  def __init__(self, clamp_output=True, **kwargs):
    """
    Initializes a Hebbian multilayer perceptron object

    Arguments:
    - clamp_output (bool, optional): if True, outputs are clamped to targets,
      if available, when computing weight updates.
    """

    self.clamp_output = clamp_output
    super().__init__(**kwargs)


  def forward(self, X, y=None):
    """
    Runs a forward pass through the network.

    Arguments:
    - X (torch.Tensor): Batch of input images.
    - y (torch.Tensor, optional): Batch of targets, stored for the backward
      pass to compute the gradients for the last layer.

    Returns:
    - y_pred (torch.Tensor): Predicted targets.
    """

    h = HebbianFunction.apply(
        X.reshape(-1, self.num_inputs),
        self.lin1.weight,
        self.lin1.bias,
        self.activation,
    )

    # if targets are provided, they can be used instead of the last layer's
    # output to train the last layer.
    if y is None or not self.clamp_output:
      targets = None
    else:
      targets = torch.nn.functional.one_hot(
          y, num_classes=self.num_outputs
          ).float()

    y_pred = HebbianFunction.apply(
        h,
        self.lin2.weight,
        self.lin2.bias,
        self.softmax,
        targets
    )

    return y_pred

#@markdown `HebbianBackpropMultiLayerPerceptron()`: Class combining Hebbian learning and backpropagation.

class HebbianBackpropMultiLayerPerceptron(MultiLayerPerceptron):
  """
  Hybrid backprop/Hebbian multilayer perceptron with one hidden layer.
  """

  def forward(self, X, y=None):
    """
    Runs a forward pass through the network.

    Arguments:
    - X (torch.Tensor): Batch of input images.
    - y (torch.Tensor, optional): Batch of targets, not used here.

    Returns:
    - y_pred (torch.Tensor): Predicted targets.
    """

    # Hebbian layer
    h = HebbianFunction.apply(
        X.reshape(-1, self.num_inputs),
        self.lin1.weight,
        self.lin1.bias,
        self.activation,
    )

    # backprop layer
    y_pred = self.softmax(self.lin2(h))

    return y_pred

class FeedbackAlignmentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                input: Tensor,
                weight: Tensor,
                bias: Optional[Tensor] = None,
                nonlinearity: Optional[Callable[[Tensor], Tensor]] = None,
                feedback_weight: Optional[Tensor] = None) -> Tensor:
        """
        Perform the forward pass for a layer using Feedback Alignment.

        Args:
            ctx: Context object to save tensors for backward pass.
            input: Input tensor.
            weight: Weight tensor.
            bias: Bias tensor (optional).
            nonlinearity: Activation function (optional).
            feedback_weight: Fixed random feedback weights (optional).

        Returns:
            output: Output tensor after linear transformation and activation.
        """
        # Compute the output: y = Wx + b
        output = input.mm(weight.t())

        # Add bias if provided
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # Apply nonlinearity if provided
        if nonlinearity is not None:
            output = nonlinearity(output)

        # Save tensors needed for backward pass
        ctx.save_for_backward(input, weight, bias, output, feedback_weight)

        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,
                 grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None, None]:
        """
        Perform the backward pass for a layer using Feedback Alignment.

        Args:
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient of the loss with respect to the output.

        Returns:
            Tuple of gradients for each input of the forward function.
        """
        # Retrieve saved tensors
        input, weight, bias, output, feedback_weight = ctx.saved_tensors

        # Initialize gradients
        grad_input = grad_weight = grad_bias = None

        # Compute input gradient using feedback weights
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(feedback_weight)

        # Compute weight gradient
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        # Compute bias gradient if bias is used
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        # Return gradients (None for nonlinearity and feedback_weight as they don't require gradients)
        return grad_input, grad_weight, grad_bias, None, None

class FeedbackAlignmentMultiLayerPerceptron(MultiLayerPerceptron):
    def __init__(self, **kwargs):
        """
        Initialize a Feedback Alignment Multi-Layer Perceptron.

        Args:
            **kwargs: Keyword arguments passed to the parent MultiLayerPerceptron class.
        """
        super().__init__(**kwargs)
        # Initialize random feedback weights for each layer
        self.feedback_weight1 = torch.nn.Parameter(torch.randn(self.num_hidden, self.num_outputs), requires_grad=False)
        self.feedback_weight2 = torch.nn.Parameter(torch.randn(self.num_outputs, self.num_hidden), requires_grad=False)

    def forward(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Perform a forward pass through the network.

        Args:
            X: Input tensor.
            y: Target tensor (not used in forward pass, included for compatibility).

        Returns:
            y_pred: Predicted output tensor.
        """
        # First layer
        h = FeedbackAlignmentFunction.apply(
            X.reshape(-1, self.num_inputs),
            self.lin1.weight,
            self.lin1.bias,
            self.activation,
            self.feedback_weight1
        )
        # Output layer
        y_pred = FeedbackAlignmentFunction.apply(
            h,
            self.lin2.weight,
            self.lin2.bias,
            self.softmax,
            self.feedback_weight2
        )
        return y_pred

class KolenPollackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                input: Tensor,
                forward_weight: Tensor,
                backward_weight: Tensor,
                bias: Optional[Tensor] = None,
                nonlinearity: Optional[Callable[[Tensor], Tensor]] = None) -> Tensor:
        output = input.mm(forward_weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        if nonlinearity is not None:
            output = nonlinearity(output)
        ctx.save_for_backward(input, forward_weight, backward_weight, bias, output)
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,
                 grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        input, forward_weight, backward_weight, bias, output = ctx.saved_tensors
        grad_input = grad_forward_weight = grad_backward_weight = grad_bias = None

        # Compute input gradient using backward weights
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(backward_weight.t())

        # Compute forward weight gradient
        if ctx.needs_input_grad[1]:
            grad_forward_weight = grad_output.t().mm(input)

        # Compute backward weight gradient
        if ctx.needs_input_grad[2]:
            grad_backward_weight = input.t().mm(grad_output)

        # Compute bias gradient if bias is used
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_forward_weight, grad_backward_weight, grad_bias, None

class KolenPollackMultiLayerPerceptron(MultiLayerPerceptron):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize separate backward weights for each layer
        self.backward_lin1 = torch.nn.Parameter(torch.randn(self.num_inputs, self.num_hidden))
        self.backward_lin2 = torch.nn.Parameter(torch.randn(self.num_hidden, self.num_outputs))

    def forward(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        # First layer
        h = KolenPollackFunction.apply(
            X.reshape(-1, self.num_inputs),
            self.lin1.weight,
            self.backward_lin1,
            self.lin1.bias,
            self.activation
        )
        # Output layer
        y_pred = KolenPollackFunction.apply(
            h,
            self.lin2.weight,
            self.backward_lin2,
            self.lin2.bias,
            self.softmax
        )
        return y_pred