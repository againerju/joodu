import math
import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F

class GaussianMixtureNLLLoss(nn.Module):
    """
    Negative log-likelihood loss over the Gaussian mixture distribution.

    """

    def __init__(self,
                 reduction: str = 'mean',
                 ) -> None:
        super(GaussianMixtureNLLLoss, self).__init__()
        
        self.reduction = reduction

        if self.reduction not in ["mean"]:
            raise ValueError("No valid reduction type {}".format(self.reduction))


    def forward(self,
                y_hat: Tensor,
                pi: Tensor,
                sigma: Tensor,
                y: Tensor,
                reg_mask: Tensor,
                is_reduce: bool = True,
                ) -> Tensor:
        """Method computes the negative log-likelihood.
        Args:
            y_hat: predicted trajectories, [K, N, T, D]
            pi: gaussian mixture coefficients, [N, K]
            sigma: standard devition, [K, N, T, 1]
            y: ground truth trajectory, [N, T, D]
            reg_mask: mask of available time steps, [N, T]
            is_reduce: indicating if the output is reduced
        Returns:
            Negative log-likelihood
        """
        # pre-process
        reg_mask = reg_mask[:, None, :, None]
        y_hat = y_hat.swapaxes(0, 1)
        sigma = sigma[:, :, :, 0].swapaxes(0, 1)
        y = torch.unsqueeze(y, 1)
        mix_coefficients = F.softmax(pi, dim=1)

        # compute the neg-log likelihood
        displacement_norms_squared = torch.sum(((y - y_hat) * reg_mask) ** 2 , dim=-1)
        normalizing_const = torch.log(2. * math.pi * sigma ** 2)

        # log-sum-exp trick: https://en.wikipedia.org/wiki/LogSumExp
        lse_args = torch.log(mix_coefficients) - torch.sum(normalizing_const + torch.divide(0.5 * displacement_norms_squared, sigma**2), axis=-1)
        max_value, _ = lse_args.max(dim=1, keepdim=True)
        ll = torch.log(torch.sum(torch.exp(lse_args - max_value), dim=-1, keepdim=True)) + max_value
        ll = torch.squeeze(ll)

        nll = -ll
        
        if is_reduce:
            if self.reduction == "mean":
                return torch.mean(nll)  
        else:
            return nll
