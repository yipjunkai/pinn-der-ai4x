import neuromancer as nm
import torch
import torch.nn as nn
import numpy as np


class DERMLP(nm.blocks.MLP):
    def __init__(self, insize, outsize, hsizes, nonlin):
        super().__init__(
            insize=insize,
            outsize=outsize * 4,
            hsizes=hsizes,
            nonlin=nonlin,
        )

        self.act = torch.nn.Softplus()

    def evidence(self, x):
        return self.act(x)

    def block_eval(self, x):
        out = super().block_eval(x).view(x.shape[0], -1, 4)

        mu, log_v, log_alpha, log_beta = [
            w.squeeze(-1) for w in torch.split(out, 1, dim=-1)
        ]

        v = self.evidence(log_v)
        alpha = self.evidence(log_alpha) + 1
        beta = self.evidence(log_beta)

        return torch.cat([mu, v, alpha, beta], dim=-1)


def NIG_NLL(diff, v, alpha, beta, name: str, scaling: int) -> nm.Constraint:
    twoBlambda = 2.0 * beta * (1.0 + v)

    res = (
        0.5 * torch.log(np.pi / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (diff) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    cons = scaling * (res == 0.0) ^ 2

    cons.update_name(name)

    return cons


def NIG_REG(diff, v, alpha, beta, name: str, scaling: int) -> nm.Constraint:
    error = torch.abs(diff)

    evi = 2 * v + alpha
    reg = error * evi

    cons = scaling * (reg == 0.0) ^ 2

    cons.update_name(name)

    return cons
