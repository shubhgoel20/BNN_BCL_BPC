import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import Prior, VariationalPosterior


class BayesianLinear(nn.Module):
    """
    Applies a linear Bayesian transformation to the incoming data: :math:`y = Ax + b`
    """

    def __init__(self, in_features, out_features, args, use_bias=True):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.device = args.device
        self.rho = args.rho

        # Variational Posterior Distributions
        self.weight_mu = nn.Parameter(
            torch.empty(
                (out_features, in_features), device=self.device, dtype=torch.float32
            ).normal_(0.0, 0.1),
            requires_grad=True,
        )
        self.weight_rho = nn.Parameter(
            self.rho
            + torch.empty(
                (out_features, in_features), device=self.device, dtype=torch.float32
            ).normal_(0.0, 0.1),
            requires_grad=True,
        )
        self.weight = VariationalPosterior(self.weight_mu, self.weight_rho, self.device)

        if self.use_bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    (out_features), device=self.device, dtype=torch.float32
                ).normal_(0.0, 0.1),
                requires_grad=True,
            )
            self.bias_rho = nn.Parameter(
                self.rho
                + nn.Parameter(
                    torch.empty(
                        out_features, device=self.device, dtype=torch.float32
                    ).normal_(0.0, 0.1),
                    requires_grad=True,
                )
            )
            self.bias = VariationalPosterior(self.bias_mu, self.bias_rho, self.device)
        else:
            self.register_parameter("bias", None)

        # Prior Distributions
        self.weight_prior = Prior(args)
        if self.use_bias:
            self.bias_prior = Prior(args)

        # Initialize log prior and log posterior
        self.log_prior = 0
        self.log_variational_posterior = 0

        self.mask_flag = False

    def prune_module(self, mask):
        self.mask_flag = True
        self.pruned_weight_mu = self.weight_mu.data.clone().mul_(mask).to(self.device)
        self.pruned_weight_rho = self.weight_rho.data.clone().mul_(mask).to(self.device)

    def calculate_logs_on_external_weight_and_bias(self, weight, bias):
        if self.use_bias:
            log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(
                bias
            )
            log_variational_posterior = self.weight.log_prob(
                weight
            ) + self.bias.log_prob(bias)
        else:
            log_prior = self.weight_prior.log_prob(weight)
            log_variational_posterior = self.weight.log_prob(weight)
        return log_prior, log_variational_posterior

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.mask_flag:
            self.weight = VariationalPosterior(
                self.pruned_weight_mu, self.pruned_weight_rho, self.device
            )

        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample() if self.use_bias else None
            self.sampled_weight = weight
            self.sampled_bias = bias
        else:
            weight = self.weight.mu
            bias = self.bias.mu if self.use_bias else None

        if self.training or calculate_log_probs:
            if self.use_bias:
                self.log_prior = self.weight_prior.log_prob(
                    weight
                ) + self.bias_prior.log_prob(bias)
                self.log_variational_posterior = self.weight.log_prob(
                    weight
                ) + self.bias.log_prob(bias)
            else:
                self.log_prior = self.weight_prior.log_prob(weight)
                self.log_variational_posterior = self.weight.log_prob(weight)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)
