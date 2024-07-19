# @title Importing dependencies

#from IPython.display import Image, SVG, display
import os
from pathlib import Path

import random
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import torchvision
import contextlib
import io

from models import *
from optimizers import *

# @markdown `download_mnist()`: Function to download MNIST.

def download_mnist(train_prop=0.8, keep_prop=0.5):

  valid_prop = 1 - train_prop

  discard_prop = 1 - keep_prop

  transform = torchvision.transforms.Compose(
      [torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.1307,), (0.3081,))]
      )


  with contextlib.redirect_stdout(io.StringIO()): #to suppress output

      full_train_set = torchvision.datasets.MNIST(
          root="./data/", train=True, download=True, transform=transform
          )
      full_test_set = torchvision.datasets.MNIST(
          root="./data/", train=False, download=True, transform=transform
          )

  train_set, valid_set, _ = torch.utils.data.random_split(
      full_train_set,
      [train_prop * keep_prop, valid_prop * keep_prop, discard_prop]
      )
  test_set, _ = torch.utils.data.random_split(
      full_test_set,
      [keep_prop, discard_prop]
      )

  print("Number of examples retained:")
  print(f"  {len(train_set)} (training)")
  print(f"  {len(valid_set)} (validation)")
  print(f"  {len(test_set)} (test)")

  return train_set, valid_set, test_set

#@markdown `get_plotting_color()`: Returns a color for the specific dataset, e.g. "train" or model index.
def get_plotting_color(dataset="train", model_idx=None):
  if model_idx is not None:
    dataset = None

  if model_idx == 0 or dataset == "train":
    color = "#1F77B4" # blue
  elif model_idx == 1 or dataset == "valid":
    color = "#FF7F0E" # orange
  elif model_idx == 2 or dataset == "test":
    color = "#2CA02C" # green
  elif model_idx == 3:
    color = "#D62728" # red
  else:
    if model_idx is not None:
      raise NotImplementedError("Colors only implemented for up to 3 models.")
    else:
      raise NotImplementedError(
          f"{dataset} dataset not recognized. Expected 'train', 'valid' "
          "or 'test'."
          )

  return color

#@markdown `plot_examples(subset)`: Plot examples from the dataset organized by their predicted class
#@markdown (if a model is provided) or by their class label otherwise
def plot_examples(subset, num_examples_per_class=8, MLP=None, seed=None, batch_size=32, num_classes=10, ax=None):
  """
  Function for visualizing example images from the dataset, organized by their
  predicted class, if a model is provided, or by their class, otherwise.

  Arguments:
  - subset (torch dataset or torch dataset subset): dataset from which to
    visualized images.
  - num_examples_per_class (int, optional): number of examples to visualize per
    class
  - MLP (MultiLayerPerceptron or None, optional): model to use to retrieve the
    predicted class for each image. If MLP is None, images will be organized by
    their class label. Otherwise, images will be organized by their predicted
    class.
  - seed (int or None, optional): Seed to use to randomly sample images to
    visualize.
  - batch_size (int, optional): If MLP is not None, number of images to
    retrieve predicted class for at one time.
  - num_classes (int, optional): Number of classes in the data.
  - ax (plt subplot, optional): Axis on which to plot images. If None, a new
    axis will be created.

  Returns:
  - ax (plt subplot): Axis on which images were plotted.
  """

  if MLP is None:
    xlabel = "Class"
  else:
    MLP.eval()
    xlabel = "Predicted class"

  if ax is None:
    fig_wid = min(8, num_classes * 0.6)
    fig_hei = min(8, num_examples_per_class * 0.6)
    _, ax = plt.subplots(figsize=(fig_wid, fig_hei))

  if seed is None:
    generator = None
  else:
    generator = torch.Generator()
    generator.manual_seed(seed)

  loader = torch.utils.data.DataLoader(
      subset, batch_size=batch_size, shuffle=True, generator=generator
      )

  plot_images = {i: list() for i in range(num_classes)}
  with torch.no_grad():
    for X, y in loader:
      if MLP is not None:
        y = MLP(X)
        y = torch.argmax(y, axis=1)

      done = True
      for i in range(num_classes):
        num_to_add = int(num_examples_per_class - len(plot_images[i]))
        if num_to_add:
          add_images = np.where(y == i)[0]
          if len(add_images):
            for add_i in add_images[: num_to_add]:
              plot_images[i].append(X[add_i, 0].numpy())
          if len(plot_images[i]) != num_examples_per_class:
            done = False

      if done:
        break

  hei, wid = X[0, 0].shape
  final_image = np.full((num_examples_per_class * hei, num_classes * wid), np.nan)
  for i, images in plot_images.items():
    if len(images):
      final_image[: len(images) * hei, i * wid: (i + 1) * wid] = np.vstack(images)

  ax.imshow(final_image, cmap="gray")

  ax.set_xlabel(xlabel)
  ax.set_xticks((np.arange(num_classes) + 0.5) * wid)
  ax.set_xticklabels([f"{int(i)}" for i in range(num_classes)])
  ax.set_yticks([])
  ax.set_title(f"Examples per {xlabel.lower()}")

  return ax

#@markdown `plot_class_distribution(train_set)`: Plots the distribution of classes in each set (train, validation, test).
def plot_class_distribution(train_set, valid_set=None, test_set=None, num_classes=10, ax=None):
  """
  Function for plotting the number of examples per class in each subset.

  Arguments:
  - train_set (torch dataset or torch dataset subset): training dataset
  - valid_set (torch dataset or torch dataset subset, optional): validation
    dataset
  - test_set (torch dataset or torch dataset subset, optional): test
    dataset
  - num_classes (int, optional): Number of classes in the data.
  - ax (plt subplot, optional): Axis on which to plot images. If None, a new
    axis will be created.

  Returns:
  - ax (plt subplot): Axis on which images were plotted.
  """

  if ax is None:
    _, ax = plt.subplots(figsize=(6, 3))

  bins = np.arange(num_classes + 1) - 0.5

  for dataset_name, dataset in [
      ("train", train_set), ("valid", valid_set), ("test", test_set)
      ]:
    if dataset is None:
      continue

    if hasattr(dataset, "dataset"):
      targets = dataset.dataset.targets[dataset.indices]
    else:
      targets = dataset.targets

    outputs = ax.hist(
        targets,
        bins=bins,
        alpha=0.3,
        color=get_plotting_color(dataset_name),
        label=dataset_name,
        )

    per_class = len(targets) / num_classes
    ax.axhline(
        per_class,
        ls="dashed",
        color=get_plotting_color(dataset_name),
        alpha=0.8
        )

  ax.set_xticks(range(num_classes))
  ax.set_title("Counts per class")
  ax.set_xlabel("Class")
  ax.set_ylabel("Count")
  ax.legend(loc="center right")

  return ax

#@markdown `plot_results(results_dict)`: Plots classification losses and
#@markdown accuracies across epochs for the training and validation sets.
def plot_results(results_dict, num_classes=10, ax=None):
  """
  Function for plotting losses and accuracies across learning.

  Arguments:
  - results_dict (dict): Dictionary storing results across epochs on training
    and validation data.
  - num_classes (float, optional): Number of classes, used to calculate chance
    accuracy.
  - ax (plt subplot, optional): Axis on which to plot results. If None, a new
    axis will be created.

  Returns:
  - ax (plt subplot): Axis on which results were plotted.
  """

  if ax is None:
    _, ax = plt.subplots(figsize=(7, 3.5))

  loss_ax = ax
  acc_ax = None

  chance = 100 / num_classes

  plotted = False
  for result_type in ["losses", "accuracies"]:
    for dataset in ["train", "valid"]:
      key = f"avg_{dataset}_{result_type}"
      if key in results_dict.keys():
        if result_type == "losses":
          ylabel = "Loss"
          plot_ax = loss_ax
          ls = None
        elif result_type == "accuracies":
          if acc_ax is None:
            acc_ax = ax.twinx()
            acc_ax.spines[["right"]].set_visible(True)
            acc_ax.axhline(chance, ls="dashed", color="k", alpha=0.8)
            acc_ax.set_ylim(-5, 105)
          ylabel = "Accuracy (%)"
          plot_ax = acc_ax
          ls = "dashed"
        else:
          raise RuntimeError(f"{result_type} result type not recognized.")

        data = results_dict[key]
        plot_ax.plot(
            data,
            ls=ls,
            label=dataset,
            alpha=0.8,
            color=get_plotting_color(dataset)
            )
        plot_ax.set_ylabel(ylabel)
        plotted = True

  if plotted:
    ax.legend(loc="center left")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f"{int(e)}" for e in range(len(data))])
    ymin, ymax = ax.get_ylim()
    if ymin > 0:
      ymin = 0
      pad = (ymax - ymin) * 0.05
      ax.set_ylim(ymin - pad, ymax + pad)

  else:
    raise RuntimeError("No data found to plot.")

  ax.set_title("Performance across learning")
  ax.set_xlabel("Epoch")

  return ax

#@markdown `plot_scores_per_class(results_dict)`: Plots the classification
#@markdown accuracies by class for the training and validation sets (for the last epoch).
def plot_scores_per_class(results_dict, num_classes=10, ax=None):
  """
  Function for plotting accuracy scores for each class.

  Arguments:
  - results_dict (dict): Dictionary storing results across epochs on training
    and validation data.
  - num_classes (int, optional): Number of classes in the data.
  - ax (plt subplot, optional): Axis on which to plot accuracies. If None, a new
    axis will be created.

  Returns:
  - ax (plt subplot): Axis on which accuracies were plotted.
  """

  if ax is None:
    _, ax = plt.subplots(figsize=(6, 3))

  avgs = list()
  ax.set_prop_cycle(None) # reset color cycle
  for s, dataset in enumerate(["train", "valid"]):
    correct_by_class = results_dict[f"{dataset}_correct_by_class"]
    seen_by_class = results_dict[f"{dataset}_seen_by_class"]
    xs, ys = list(), list()
    for i, total in seen_by_class.items():
      xs.append(i + 0.3 * (s - 0.5))
      if total == 0:
        ys.append(np.nan)
      else:
        ys.append(100 * correct_by_class[i] / total)

    avg_key = f"avg_{dataset}_accuracies"
    if avg_key in results_dict.keys():
      ax.axhline(
          results_dict[avg_key][-1], ls="dashed", alpha=0.8,
          color=get_plotting_color(dataset)
          )

    ax.bar(
        xs, ys, label=dataset, width=0.3, alpha=0.8,
        color=get_plotting_color(dataset)
        )

  ax.set_xticks(range(num_classes))
  ax.set_xlabel("Class")
  ax.set_ylabel("Accuracy (%)")
  ax.set_title("Class scores")
  ax.set_ylim(-5, 105)

  chance = 100 / num_classes
  ax.axhline(chance, ls="dashed", color="k", alpha=0.8)

  ax.legend()

  return ax


#@markdown `plot_weights(MLP)`: Plots weights before and after training.
def plot_weights(MLP, shared_colorbar=False):
  """
  Function for plotting model weights and biases before and after learning.

  Arguments:
  - MLP (torch model): Model for which to plot weights and biases.
  - shared_colorbar (bool, optional): If True, one colorbar is shared for all
      parameters.

  Returns:
  - ax (plt subplot array): Axes on which weights and biases were plotted.
  """

  param_names = MLP.list_parameters()

  params_images = dict()
  pre_means = dict()
  post_means = dict()
  vmin, vmax = np.inf, -np.inf
  for param_name in param_names:
    layer, param_type = param_name.split("_")
    init_params = getattr(MLP, f"init_{layer}_{param_type}").numpy()
    separator = np.full((1, init_params.shape[-1]), np.nan)
    last_params = getattr(getattr(MLP, layer), param_type).detach().numpy()
    diff_params = last_params - init_params

    params_image = np.vstack(
        [init_params, separator, last_params, separator, diff_params]
        )
    vmin = min(vmin, np.nanmin(params_image))
    vmax = min(vmax, np.nanmax(params_image))

    params_images[param_name] = params_image
    pre_means[param_name] = init_params.mean()
    post_means[param_name] = last_params.mean()

  nrows = len(param_names)
  gridspec_kw = dict()
  if len(param_names) == 4:
    gridspec_kw["height_ratios"] = [5, 1, 5, 1]
    cbar_label = "Weight/bias values"
  elif len(param_names) == 2:
    gridspec_kw["height_ratios"] = [5, 5]
    cbar_label = "Weight values"
  else:
    raise NotImplementedError("Expected 2 parameters (weights only) or "
      f"4 parameters (weights and biases), but found {len(param_names)}"
    )

  if shared_colorbar:
    nrows += 1
    gridspec_kw["height_ratios"].append(1)
  else:
    vmin, vmax = None, None

  fig, axes = plt.subplots(
      nrows, 1, figsize=(6, nrows + 3), gridspec_kw=gridspec_kw
      )

  for i, (param_name, params_image) in enumerate(params_images.items()):
    layer, param_type = param_name.split("_")
    layer_str = "First" if layer == "lin1" else "Second"
    param_str = "weights" if param_type == "weight" else "biases"

    axes[i].set_title(f"{layer_str} linear layer {param_str} (pre, post and diff)")
    im = axes[i].imshow(params_image, aspect="auto", vmin=vmin, vmax=vmax)
    if not shared_colorbar:
      cbar = fig.colorbar(im, ax=axes[i], aspect=10)
      cbar.ax.axhline(pre_means[param_name], ls="dotted", color="k", alpha=0.5)
      cbar.ax.axhline(post_means[param_name], color="k", alpha=0.5)

    if param_type == "weight":
      axes[i].set_xlabel("Input dim.")
      axes[i].set_ylabel("Output dim.")
    axes[i].spines[["left", "bottom"]].set_visible(False)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

  if shared_colorbar:
    cax = axes[-1]
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", location="bottom")
    cax.set_xlabel(cbar_label)

  return axes

# Training

#@markdown `train_model(MLP, train_loader, valid_loader, optimizer)`: Main function.
#@markdown Trains the model across epochs. Aggregates loss and accuracy statistics
#@markdown from the training and validation datasets into a results dictionary which is returned.

def train_model(MLP, train_loader, valid_loader, optimizer, num_epochs=5):
  """
  Train a model for several epochs.

  Arguments:
  - MLP (torch model): Model to train.
  - train_loader (torch dataloader): Dataloader to use to train the model.
  - valid_loader (torch dataloader): Dataloader to use to validate the model.
  - optimizer (torch optimizer): Optimizer to use to update the model.
  - num_epochs (int, optional): Number of epochs to train model.

  Returns:
  - results_dict (dict): Dictionary storing results across epochs on training
    and validation data.
  """

  results_dict = {
      "avg_train_losses": list(),
      "avg_valid_losses": list(),
      "avg_train_accuracies": list(),
      "avg_valid_accuracies": list(),
  }

  for e in tqdm(range(num_epochs)):
    no_train = True if e == 0 else False # to get a baseline
    latest_epoch_results_dict = train_epoch(
        MLP, train_loader, valid_loader, optimizer=optimizer, no_train=no_train
        )

    for key, result in latest_epoch_results_dict.items():
      if key in results_dict.keys() and isinstance(results_dict[key], list):
        results_dict[key].append(latest_epoch_results_dict[key])
      else:
        results_dict[key] = result # copy latest

  return results_dict


def train_epoch(MLP, train_loader, valid_loader, optimizer, no_train=False):
  """
  Train a model for one epoch.

  Arguments:
  - MLP (torch model): Model to train.
  - train_loader (torch dataloader): Dataloader to use to train the model.
  - valid_loader (torch dataloader): Dataloader to use to validate the model.
  - optimizer (torch optimizer): Optimizer to use to update the model.
  - no_train (bool, optional): If True, the model is not trained for the
    current epoch. Allows a baseline (chance) performance to be computed in the
    first epoch before training starts.

  Returns:
  - epoch_results_dict (dict): Dictionary storing epoch results on training
    and validation data.
  """

  criterion = torch.nn.NLLLoss()

  epoch_results_dict = dict()
  for dataset in ["train", "valid"]:
    for sub_str in ["correct_by_class", "seen_by_class"]:
      epoch_results_dict[f"{dataset}_{sub_str}"] = {
          i:0 for i in range(MLP.num_outputs)
          }

  MLP.train()
  train_losses, train_acc = list(), list()
  for X, y in train_loader:
    y_pred = MLP(X, y=y)
    loss = criterion(torch.log(y_pred), y)
    acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
    train_losses.append(loss.item() * len(y))
    train_acc.append(acc.item() * len(y))
    update_results_by_class_in_place(
        y, y_pred.detach(), epoch_results_dict, dataset="train",
        num_classes=MLP.num_outputs
        )
    optimizer.zero_grad()
    if not no_train:
      loss.backward()
      optimizer.step()

  num_items = len(train_loader.dataset)
  epoch_results_dict["avg_train_losses"] = np.sum(train_losses) / num_items
  epoch_results_dict["avg_train_accuracies"] = np.sum(train_acc) / num_items * 100

  MLP.eval()
  valid_losses, valid_acc = list(), list()
  with torch.no_grad():
    for X, y in valid_loader:
      y_pred = MLP(X)
      loss = criterion(torch.log(y_pred), y)
      acc = (torch.argmax(y_pred, axis=1) == y).sum() / len(y)
      valid_losses.append(loss.item() * len(y))
      valid_acc.append(acc.item() * len(y))
      update_results_by_class_in_place(
          y, y_pred.detach(), epoch_results_dict, dataset="valid"
          )

  num_items = len(valid_loader.dataset)
  epoch_results_dict["avg_valid_losses"] = np.sum(valid_losses) / num_items
  epoch_results_dict["avg_valid_accuracies"] = np.sum(valid_acc) / num_items * 100

  return epoch_results_dict


def update_results_by_class_in_place(y, y_pred, result_dict, dataset="train",
                                     num_classes=10):
  """
  Updates results dictionary in place during a training epoch by adding data
  needed to compute the accuracies for each class.

  Arguments:
  - y (torch Tensor): target labels
  - y_pred (torch Tensor): predicted targets
  - result_dict (dict): Dictionary storing epoch results on training
    and validation data.
  - dataset (str, optional): Dataset for which results are being added.
  - num_classes (int, optional): Number of classes.
  """

  correct_by_class = None
  seen_by_class = None

  y_pred = np.argmax(y_pred, axis=1)
  if len(y) != len(y_pred):
    raise RuntimeError("Number of predictions does not match number of targets.")

  for i in result_dict[f"{dataset}_seen_by_class"].keys():
    idxs = np.where(y == int(i))[0]
    result_dict[f"{dataset}_seen_by_class"][int(i)] += len(idxs)

    num_correct = int(sum(y[idxs] == y_pred[idxs]))
    result_dict[f"{dataset}_correct_by_class"][int(i)] += num_correct

#@markdown The following function returns a dataset restricted to specific classes:

#@markdown `restrict_classes(dataset)`: Keeps specified classes in a dataset.

def restrict_classes(dataset, classes=[6], keep=True):
  """
  Removes or keeps specified classes in a dataset.

  Arguments:
  - dataset (torch dataset or subset): Dataset with class targets.
  - classes (list): List of classes to keep or remove.
  - keep (bool): If True, the classes specified are kept. If False, they are
  removed.

  Returns:
  - new_dataset (torch dataset or subset): Datset restricted as specified.
  """

  if hasattr(dataset, "dataset"):
    indices = np.asarray(dataset.indices)
    targets = dataset.dataset.targets[indices]
    dataset = dataset.dataset
  else:
    indices = np.arange(len(dataset))
    targets = dataset.targets

  specified_idxs = np.isin(targets, np.asarray(classes))
  if keep:
    retain_indices = indices[specified_idxs]
  else:
    retain_indices = indices[~specified_idxs]

  new_dataset = torch.utils.data.Subset(dataset, retain_indices)

  return new_dataset

#@markdown `train_model_extended`: Initializes model and optimizer, restricts datasets to
#@markdown specified classes, trains model. Returns trained model and results dictionary.

def train_model_extended(model_type="backprop", keep_num_classes="all", lr=0.1,
                         num_epochs=5, partial_backprop=False, num_inputs=784,
                         num_hidden=100, bias=0.0,
                         batch_size=32, plot_distribution=False, train_set=None, valid_set=None, test_set=None):
  """
  Initializes model and optimizer, restricts datasets to specified classes and
  trains the model. Returns the trained model and results dictionary.

  Arguments:
  - model_type (str, optional): model to initialize ("backprop" or "Hebbian")
  - keep_num_classes (str or int, optional): number of classes to keep (from 0)
  - lr (float or list, optional): learning rate for both or each layer
  - num_epochs (int, optional): number of epochs to train model.
  - partial_backprop (bool, optional): if True, backprop is used to train the
    final Hebbian learning model.
  - num_hidden (int, optional): number of hidden units in the hidden layer
  - bias (bool, optional): if True, each linear layer will have biases in
      addition to weights.
  - batch_size (int, optional): batch size for dataloaders.
  - plot_distribution (bool, optional): if True, dataset class distributions
    are plotted.

  Returns:
  - MLP (torch module): Model
  - results_dict (dict): Dictionary storing results across epochs on training
    and validation data.
  """

  if isinstance(keep_num_classes, str):
    if keep_num_classes == "all":
      num_classes = 10
      use_train_set = train_set
      use_valid_set = valid_set
    else:
      raise ValueError("If 'keep_classes' is a string, it should be 'all'.")
  else:
    num_classes = int(keep_num_classes)
    use_train_set = restrict_classes(train_set, np.arange(keep_num_classes))
    use_valid_set = restrict_classes(valid_set, np.arange(keep_num_classes))

  if plot_distribution:
    plot_class_distribution(use_train_set, use_valid_set)

  train_loader = torch.utils.data.DataLoader(
      use_train_set, batch_size=batch_size, shuffle=True
      )
  valid_loader = torch.utils.data.DataLoader(
      use_valid_set, batch_size=batch_size, shuffle=False
        )

  model_params = {
      "num_inputs": num_inputs,
      "num_hidden": num_hidden,
      "num_outputs": num_classes,
      "bias": bias,
  }

  if model_type.lower() == "backprop":
    Model = MultiLayerPerceptron
  elif model_type.lower() == "hebbian":
    if partial_backprop:
      Model = HebbianBackpropMultiLayerPerceptron
    else:
      Model = HebbianMultiLayerPerceptron
      model_params["clamp_output"] = True
  else:
    raise ValueError(
        f"Got {model_type} model type, but expected 'backprop' or 'hebbian'."
        )

  MLP = Model(**model_params)

  if isinstance(lr, list):
    if len(lr) != 2:
      raise ValueError("If 'lr' is a list, it must be of length 2.")
    optimizer = BasicOptimizer([
        {"params": MLP.lin1.parameters(), "lr": lr[0]},
        {"params": MLP.lin2.parameters(), "lr": lr[1]},
    ])
  else:
    optimizer = BasicOptimizer(MLP.parameters(), lr=lr)


  results_dict = train_model(
      MLP,
      train_loader,
      valid_loader,
      optimizer,
      num_epochs=num_epochs
      )

  return MLP, results_dict

#@markdown `compute_gradient_SNR(MLP, dataset)`: Passes a dataset through a model
#@markdown and computes the SNR of the gradients computed by the model for each example.
#@markdown Returns a dictionary containing the gradient SNRs for each layer of a model.

def compute_gradient_SNR(MLP, dataset):
  """
  Computes gradient SNRs for a model given a dataset.

  Arguments:
  - MLP (torch model): Model for which to compute gradient SNRs.
  - dataset (torch dataset): Dataset with which to compute gradient SNRs.

  Returns:
  - SNR_dict (dict): Dictionary compiling gradient SNRs for each parameter
    (i.e., the weights and/or biases of each layer).
  """

  MLP.eval()

  criterion = torch.nn.NLLLoss()

  gradients = {key: list() for key in MLP.list_parameters()}

  # initialize a loader with a batch size of 1
  loader = torch.utils.data.DataLoader(dataset, batch_size=1)

  # collect gradients computed on the dataset of data
  for X, y in loader:
    y_pred = MLP(X, y=y)
    loss = criterion(torch.log(y_pred), y)

    MLP.zero_grad() # zero grad before
    loss.backward()
    for key, value in MLP.gather_gradient_dict().items():
      gradients[key].append(value)
    MLP.zero_grad() # zero grad after, since no optimzer step is taken

  # aggregate the gradients
  SNR_dict = {key: list() for key in MLP.list_parameters()}
  for key, value in gradients.items():
      SNR_dict[key] = compute_SNR(np.asarray(value))

  return SNR_dict


def compute_SNR(data, epsilon=1e-7):
    """
    Calculates the average SNR for of data across the first axis.

    Arguments:
    - data (torch Tensor): items x gradients
    - epsilon (float, optional): value added to the denominator to avoid
      division by zero.

    Returns:
    - avg_SNR (float): average SNR across data items
    """

    absolute_mean = np.abs(np.mean(data, axis=0))
    std = np.std(data, axis=0)
    SNR_by_item = absolute_mean / (std + epsilon)
    avg_SNR = np.mean(SNR_by_item)

    return avg_SNR

#@markdown `plot_gradient_SNRs(SNR_dict)`: Plots gradient SNRs collected in
#@markdown a dictionary.
def plot_gradient_SNRs(SNR_dict, width=0.5, ax=None):
  """
  Plot gradient SNRs for various learning rules.

  Arguments:
  - SNR_dict (dict): Gradient SNRs for each learning rule.
  - width (float, optional): Width of the bars.
  - ax (plt subplot, optional): Axis on which to plot gradient SNRs. If None, a
    new axis will be created.

  Returns:
  - ax (plt subplot): Axis on which gradient SNRs were plotted.  """

  if ax is None:
    wid = min(8, len(SNR_dict) * 1.5)
    _, ax = plt.subplots(figsize=(wid, 4))

  xlabels = list()
  SNR_means = list()
  SNR_sems = list()
  SNRs_scatter = list()
  for m, (model_type, SNRs) in enumerate(SNR_dict.items()):
    xlabels.append(model_type)
    color = get_plotting_color(model_idx=m)
    ax.bar(
        m, np.mean(SNRs), yerr=scipy.stats.sem(SNRs),
        alpha=0.5, width=width, capsize=5, color=color
        )
    s = [20 + i * 30 for i in range(len(SNRs))]
    ax.scatter([m] * len(SNRs), SNRs, alpha=0.8, s=s, color=color, zorder=5)

  x = np.arange(len(xlabels))
  ax.set_xticks(x)
  x_pad = (x.max() - x.min() + width) * 0.3
  ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
  ax.set_xticklabels(xlabels, rotation=45)
  ax.set_xlabel("Learning rule")
  ax.set_ylabel("SNR")
  ax.set_title("SNR of the gradients")

  return ax

#@markdown `train_and_calculate_cosine_sim(MLP, train_loader, valid_loader, optimizer)`:
#@markdown Trains a model using a specific learning rule, while computing the cosine
#@markdown similarity of the gradients proposed the learning rule compared to those proposed
#@markdown by error backpropagation.
def train_and_calculate_cosine_sim(MLP, train_loader, valid_loader, optimizer,
                                   num_epochs=8):
  """
  Train model across epochs, calculating the cosine similarity between the
  gradients proposed by the learning rule it's trained with, compared to those
  proposed by error backpropagation.

  Arguments:
  - MLP (torch model): Model to train.
  - train_loader (torch dataloader): Dataloader to use to train the model.
  - valid_loader (torch dataloader): Dataloader to use to validate the model.
  - optimizer (torch optimizer): Optimizer to use to update the model.
  - num_epochs (int, optional): Number of epochs to train model.

  Returns:
  - cosine_sim (dict): Dictionary storing the cosine similarity between the
    model's learning rule and backprop across epochs, computed on the
    validation data.
  """

  criterion = torch.nn.NLLLoss()

  cosine_sim_dict = {key: list() for key in MLP.list_parameters()}

  for e in tqdm(range(num_epochs)):
    MLP.train()
    for X, y in train_loader:
      y_pred = MLP(X, y=y)
      loss = criterion(torch.log(y_pred), y)
      optimizer.zero_grad()
      if e != 0:
        loss.backward()
        optimizer.step()

    MLP.eval()
    lr_gradients_dict = {key: list() for key in cosine_sim_dict.keys()}
    backprop_gradients_dict = {key: list() for key in cosine_sim_dict.keys()}

    for X, y in valid_loader:
      # collect gradients computed with learning rule
      y_pred = MLP(X, y=y)
      loss = criterion(torch.log(y_pred), y)

      MLP.zero_grad()
      loss.backward()
      for key, value in MLP.gather_gradient_dict().items():
        lr_gradients_dict[key].append(value)
      MLP.zero_grad()

      # collect gradients computed with backprop
      y_pred = MLP.forward_backprop(X)
      loss = criterion(torch.log(y_pred), y)

      MLP.zero_grad()
      loss.backward()
      for key, value in MLP.gather_gradient_dict().items():
        backprop_gradients_dict[key].append(value)
      MLP.zero_grad()

    for key in cosine_sim_dict.keys():
      lr_grad = np.asarray(lr_gradients_dict[key])
      bp_grad = np.asarray(backprop_gradients_dict[key])
      if (lr_grad == 0).all():
        warnings.warn(
          f"Learning rule computed all 0 gradients for epoch {e}. "
          "Cosine similarity cannot be calculated."
          )
        epoch_cosine_sim = np.nan
      elif (bp_grad == 0).all():
        warnings.warn(
          f"Backprop. rule computed all 0 gradients for epoch {e}. "
          "Cosine similarity cannot be calculated."
          )
        epoch_cosine_sim = np.nan
      else:
        epoch_cosine_sim = calculate_cosine_similarity(lr_grad, bp_grad)

      cosine_sim_dict[key].append(epoch_cosine_sim)

  return cosine_sim_dict


def calculate_cosine_similarity(data1, data2):
    """
    Calculates the cosine similarity between two vectors.

    Arguments:
    - data1 (torch Tensor): first vector
    - data2 (torch Tensor): second vector
    """

    data1 = data1.reshape(-1)
    data2 = data2.reshape(-1)

    numerator = np.dot(data1, data2)
    denominator = (
        np.sqrt(np.dot(data1, data1)) * np.sqrt(np.dot(data2, data2))
      )

    cosine_sim = numerator / denominator

    return cosine_sim

#@markdown `plot_gradient_cosine_sims(cosine_sim_dict)`: Plots cosine similarity
#@markdown of the gradients proposed by a model across learning to those proposed
#@markdown by error backpropagation.
def plot_gradient_cosine_sims(cosine_sim_dict, ax=None):
  """
  Plot gradient cosine similarities to error backpropagation for various
  learning rules.

  Arguments:
  - cosine_sim_dict (dict): Gradient cosine similarities for each learning rule.
  - ax (plt subplot, optional): Axis on which to plot gradient cosine
    similarities. If None, a new axis will be created.

  Returns:
  - ax (plt subplot): Axis on which gradient cosine similarities were plotted.
  """

  if ax is None:
    _, ax = plt.subplots(figsize=(8, 4))

  max_num_epochs = 0
  for m, (model_type, cosine_sims) in enumerate(cosine_sim_dict.items()):
    cosine_sims = np.asarray(cosine_sims) # params x epochs
    num_epochs = cosine_sims.shape[1]
    x = np.arange(num_epochs)
    cosine_sim_means = np.nanmean(cosine_sims, axis=0)
    cosine_sim_sems = scipy.stats.sem(cosine_sims, axis=0, nan_policy="omit")

    ax.plot(x, cosine_sim_means, label=model_type, alpha=0.8)

    color = get_plotting_color(model_idx=m)
    ax.fill_between(
        x,
        cosine_sim_means - cosine_sim_sems,
        cosine_sim_means + cosine_sim_sems,
        alpha=0.3, lw=0, color=color
        )

    for i, param_cosine_sims in enumerate(cosine_sims):
      s = 20 + i * 30
      ax.scatter(x, param_cosine_sims, color=color, s=s, alpha=0.6)

    max_num_epochs = max(max_num_epochs, num_epochs)

  if max_num_epochs > 0:
    x = np.arange(max_num_epochs)
    xlabels = [f"{int(e)}" for e in x]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

  ymin = ax.get_ylim()[0]
  ymin = min(-0.1, ymin)
  ax.set_ylim(ymin, 1.1)

  ax.axhline(0, ls="dashed", color="k", zorder=-5, alpha=0.5)

  ax.set_xlabel("Epoch")
  ax.set_ylabel("Cosine similarity")
  ax.set_title("Cosine similarity to backprop gradients")
  ax.legend()

  return ax