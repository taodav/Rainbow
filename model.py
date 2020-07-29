# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class LinearHead(nn.Module):
  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
    super().__init__()
    self.d1 = nn.Linear(input_dim, hidden_dim)
    self.d2 = nn.Linear(hidden_dim, output_dim)
    self.bn = nn.BatchNorm1d(hidden_dim)

  def forward(self, x):
    x = x.reshape((x.shape[0], -1))
    x = self.d1(x)
    if x.shape[0] > 1:
      x = self.bn(x)
    x = F.relu(x)
    x = self.d2(x)

    return x

# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input, only_mu=False):
    if self.training or not only_mu:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class MPREncoder(nn.Module):
  def __init__(self, frames: int, dropout: float = 0.5):
    super().__init__()
    channels=[32, 64, 64]
    kernel_sizes=[8, 4, 3]
    strides=[4, 2, 1]
    paddings=[0, 0, 0]
    nonlinearity=nn.ReLU
    dropout_layer = nn.Dropout(dropout)
    in_channels = [frames] + channels[:-1]
    conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
                                   kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
                   zip(in_channels, channels, kernel_sizes, strides, paddings)]
    sequence = list()
    for conv_layer in conv_layers:
      if dropout > 0.:
        sequence.extend([conv_layer, nonlinearity(), dropout_layer])
      else:
        sequence.extend([conv_layer, nonlinearity()])
    self.conv = torch.nn.Sequential(*sequence)

  def forward(self, x):
    return self.conv(x)


class MPRTransitionModel(nn.Module):
  def __init__(self, n_actions: int):
    super().__init__()
    self.l1 = nn.Conv2d(64 + n_actions, 64, 3, padding=1)
    self.bn = nn.BatchNorm2d(64)
    self.l2 = nn.Conv2d(64, 64, 3, padding=1)

  def forward(self, x):
    x = self.l1(x)
    if x.shape[0] > 1:
      x = self.bn(x)
    x = F.relu(x)
    x = F.relu(self.l2(x))
    return x


class MPR(nn.Module):
  def __init__(self, frames: int,
               n_actions: int, repr_dim: int = 64 * 7 * 7,
               proj_dim: int = 256, augmentation: bool = True):

    super().__init__()

    self.m = 0.99
    self.f = frames
    self.n_actions = n_actions
    self._proj_dim = proj_dim

    # Online network
    self.encoder = MPREncoder(frames, dropout=0.5 if augmentation else 0.)
    self.transition = MPRTransitionModel(n_actions)
    self.projector = LinearHead(repr_dim, proj_dim, proj_dim)
    self.predictor = LinearHead(proj_dim, 2*proj_dim, proj_dim)

    # Target network
    self.encoder_t = MPREncoder(frames)
    self.projector_t = LinearHead(repr_dim, proj_dim, proj_dim)

    self.augmentation = augmentation

    for param_t in self.encoder_t.parameters():
      param_t.requires_grad = False  # not update by gradient
    for param_t in self.projector_t.parameters():
      param_t.requires_grad = False  # not update by gradient

  @property
  def proj_dim(self):
    return self._proj_dim

  @torch.no_grad()
  def _update_target_network(self):
    for param_q, param_t in zip(self.encoder.parameters(), self.encoder_t.parameters()):
      param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)
    for param_q, param_t in zip(self.projector.parameters(), self.projector_t.parameters()):
      param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)

  @torch.no_grad()
  def _one_hot_actions(self, actions: torch.Tensor) -> torch.Tensor:

    bs, ts = actions.shape  # batch size, timesteps (frames + k_steps)
    actions = actions[..., None, None, None]  # Add CxHxW dimensions
    actions = actions.expand(bs, ts, 1, 7, 7).long()
    one_hot = torch.zeros(bs, ts, self.n_actions, 7, 7, device=actions.device)
    one_hot = one_hot.scatter_(2, actions, 1)
    return one_hot

  def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
    # bs, ts, _, _ = observations.shape
    bs, ts = actions.shape
    k_steps = ts - self.f

    losses = []
    target = self._one_hot_actions(actions)

    im_o = observations[:,:self.f]
    q_e = self.encoder(im_o)

    for k in range(1, k_steps + 1):
      # compute online features
      q_e = torch.cat((q_e, target[:,self.f-1+k-1]), 1)
      q_e = self.transition(q_e)
      q = self.projector(q_e)
      q = self.predictor(q)

      # compute target features
      with torch.no_grad():
        im_t = observations[:,k:k+self.f]
        k_e = self.encoder_t(im_t)
        k = self.projector_t(k_e)

      losses.append(-2 * F.cosine_similarity(q, k).mean())

    with torch.no_grad():
      self._update_target_network()

    return losses

class MPRDQN(nn.Module):
  def __init__(self, args, action_space):
    super(MPRDQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    self.mpr = MPR(args.history_length, n_actions=action_space)

    self.fc_h_v = NoisyLinear(self.mpr.proj_dim, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.mpr.proj_dim, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    # x = self.convs(x)
    # x = x.view(-1, self.conv_output_size)
    x = self.mpr.encoder(x)
    x = F.relu(self.mpr.projector(x))

    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()
