from pinn_der_ai4x import PINNTrainer, DERMLP, NIG_REG, NIG_NLL
import neuromancer as nm
import torch
import numpy as np


class PINNTrainer(PINNTrainer):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    def define_neural_network(self):
        input_size = 2
        output_size = 1
        hidden_sizes = [32, 32]

        return DERMLP(
            insize=input_size,
            outsize=output_size,
            hsizes=hidden_sizes,
            nonlin=torch.nn.Tanh,
        )

    def define_decision_variables(self):

        x_var = nm.variable("x")
        t_var = nm.variable("t")

        # U changed to Mean, Variance, Evidence (nu), Log evidence (alpha)
        u = nm.variable("u")
        u_mean_var = u[:, [0]]
        u_v_var = u[:, [1]]
        u_alpha_var = u[:, [2]]
        u_beta_var = u[:, [3]]

        return {
            "x": x_var,
            "t": t_var,
            "u_mean": u_mean_var,
            "u_v": u_v_var,
            "u_alpha": u_alpha_var,
            "u_beta": u_beta_var,
        }

    def define_residual_pde(self, decision_vars):

        x, t, u_mean = decision_vars["x"], decision_vars["t"], decision_vars["u_mean"]

        nu = 0.01 / np.pi

        du_dt = u_mean.grad(t)
        du_dx = u_mean.grad(x)
        d2u_dx2 = du_dx.grad(x)

        f_pinn = du_dt + u_mean * du_dx - nu * d2u_dx2

        return {
            "f_pinn": f_pinn,
        }

    def define_objective_function(
        self,
        decision_vars,
        residual_pde,
    ):
        scaling = 100.0  # Scaling factor for better convergence
        Nu = 200  # Number of initial conditions

        f_pinn = residual_pde["f_pinn"]

        u_mean, u_v, u_alpha, u_beta = (
            decision_vars["u_mean"],
            decision_vars["u_v"],
            decision_vars["u_alpha"],
            decision_vars["u_beta"],
        )

        # Extract training data
        Y_train_Nu = self.Y_train_Nu

        REG_WEIGHT = 0.0001

        metric_1 = f_pinn
        metric_2 = u_mean[-Nu:] - Y_train_Nu

        # Evidential loss
        nig_nll_1 = NIG_NLL(
            metric_1,
            u_v,
            u_alpha,
            u_beta,
            "nig_nll_1",
            scaling,
        )

        nig_reg_1 = NIG_REG(
            metric_1,
            u_v,
            u_alpha,
            u_beta,
            "nig_reg_1",
            scaling * REG_WEIGHT,
        )

        nig_nll_2 = NIG_NLL(
            metric_2,
            u_v[-Nu:],
            u_alpha[-Nu:],
            u_beta[-Nu:],
            "nig_nll_2",
            scaling,
        )

        nig_reg_2 = NIG_REG(
            metric_2,
            u_v[-Nu:],
            u_alpha[-Nu:],
            u_beta[-Nu:],
            "nig_reg_2",
            scaling * REG_WEIGHT,
        )

        return [nig_nll_1, nig_reg_1, nig_nll_2, nig_reg_2]

    def define_constraints(self, decision_vars):
        scaling = 100.0

        u_mean, u_v, u_alpha, u_beta = (
            decision_vars["u_mean"],
            decision_vars["u_v"],
            decision_vars["u_alpha"],
            decision_vars["u_beta"],
        )

        # Output constraints to bound the PINN solution in the PDE output domain [-1.0, 1.0]
        con_1 = scaling * (u_mean <= 1.0) ^ 2
        con_2 = scaling * (u_mean >= -1.0) ^ 2

        con_1.update_name("con_1")
        con_2.update_name("con_2")

        return [con_1, con_2]
