from pinn_der_ai4x import PINNTrainer, DERMLP, NIG_REG, NIG_NLL
import neuromancer as nm
import torch
import torch.nn as nn
import numpy as np


class PINNTrainer(PINNTrainer):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    def define_neural_network(self):
        input_size = 2
        output_size = 1
        hidden_sizes = [30, 30]

        return DERMLP(
            insize=input_size,
            outsize=output_size,
            hsizes=hidden_sizes,
            nonlin=torch.nn.SiLU,
        )

    def define_decision_variables(self):

        x_var = nm.variable("x")
        y_var = nm.variable("y")

        T_var = nm.variable("T")
        T_mean_var = T_var[:, [0]]
        T_v_var = T_var[:, [1]]
        T_alpha_var = T_var[:, [2]]
        T_beta_var = T_var[:, [3]]

        return {
            "x": x_var,
            "y": y_var,
            "T_mean": T_mean_var,
            "T_v": T_v_var,
            "T_alpha": T_alpha_var,
            "T_beta": T_beta_var,
        }

    def define_residual_pde(self, decision_vars):

        x, y, T = decision_vars["x"], decision_vars["y"], decision_vars["T_mean"]

        dT_dx = T.grad(x)
        dT_dy = T.grad(y)
        d2T_dx2 = dT_dx.grad(x)
        d2T_dy2 = dT_dy.grad(y)

        f_pinn = d2T_dx2 + d2T_dy2

        return {
            "f_pinn": f_pinn,
        }

    def define_objective_function(
        self,
        decision_vars,
        residual_pde,
    ):
        f_pinn = residual_pde["f_pinn"]

        x, y, T_mean, T_v, T_alpha, T_beta = (
            decision_vars["x"],
            decision_vars["y"],
            decision_vars["T_mean"],
            decision_vars["T_v"],
            decision_vars["T_alpha"],
            decision_vars["T_beta"],
        )

        N_bc = self.N_bc
        T_train_bc = self.T_train_bc

        # scaling factor for better convergence
        scaling_cp = 1.0
        scaling_bc = 1.0

        # PDE CP loss (MSE)
        metric_1 = f_pinn

        # PDE BC loss (MSE)
        metric_2 = T_mean[-N_bc:] - T_train_bc

        REG_WEIGHT = 0.0001

        nig_nll_1 = NIG_NLL(metric_1, T_v, T_alpha, T_beta, "nig_nll_1", scaling_cp)

        nig_reg_1 = NIG_REG(
            metric_1,
            T_v,
            T_alpha,
            T_beta,
            "nig_reg_1",
            scaling_cp * REG_WEIGHT,
        )

        nig_nll_2 = NIG_NLL(
            metric_2,
            T_v[-N_bc:],
            T_alpha[-N_bc:],
            T_beta[-N_bc:],
            "nig_nll_2",
            scaling_bc,
        )

        nig_reg_2 = NIG_REG(
            metric_2,
            T_v[-N_bc:],
            T_alpha[-N_bc:],
            T_beta[-N_bc:],
            "nig_reg_2",
            scaling_bc * REG_WEIGHT,
        )

        return [
            nig_nll_1,
            nig_reg_1,
            nig_nll_2,
            nig_reg_2,
        ]

    def define_constraints(self, decision_vars):

        x, y, T = decision_vars["x"], decision_vars["y"], decision_vars["T_mean"]

        # output constraints to bound the PINN solution
        con_1 = T >= 0  # Passive scalar T is always positive; note: this is optional

        return [con_1]
