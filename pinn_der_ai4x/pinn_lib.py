import torch
import neuromancer as nm
from neuromancer.trainer import Trainer
import logging
from neuromancer.callbacks import Callback
from neuromancer.loggers import BasicLogger
from tqdm import tqdm
from typing import Dict, List, Any
from neuromancer.constraint import Variable, Objective, Constraint
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import numpy as np


class PINNTrainer(ABC):
    def __init__(self, params: Dict[str, Any], **kwargs):
        """
        Initialize the PINN trainer with parameters.

        Args:
            params (dict): Dictionary of parameters.
            **kwargs: Additional keyword arguments. These are passed to the class variables.
        """
        self.logger = logging.getLogger(__name__)
        self.params = params
        self.model = None

        # Handle any additional keyword arguments
        # Expect type of kwargs to be Dict[str, Any]
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def define_neural_network(self) -> torch.nn.Module:
        """
        Define the neural network architecture.
        This method should be implemented by the user.

        Returns:
            torch.nn.Module: The neural network model.
        """
        pass

    @abstractmethod
    def define_decision_variables(self) -> Dict[str, Variable]:
        """
        Define the decision variables for the optimization problem.
        This method should be implemented by the user.

        Returns:
            dict: A dictionary of decision variables.
        """
        pass

    @abstractmethod
    def define_residual_pde(
        self, decision_vars: Dict[str, Variable]
    ) -> Variable | Dict[str, Variable]:
        """
        Define the residual PDE loss for the PINN.

        Args:
            decision_vars (dict): Dictionary of decision variables.

        Returns:
            nm.Variable: The residual PDE loss.
        """
        pass

    @abstractmethod
    def define_objective_function(
        self,
        decision_vars: Dict[str, Variable],
        residual_pde: Variable | Dict[str, Variable],
        training_loader: DataLoader,
    ) -> List[Objective]:
        """
        Define the objective function for the PINN.
        This method should be implemented by the user.

        Args:
            decision_vars (dict): Dictionary of decision variables.
            residual_pde (nm.Variable): The residual PDE constraints.
            training_loader (DataLoader): DataLoader for training data.

        Returns:
            list: A list of objectives.
        """
        pass

    @abstractmethod
    def define_constraints(self, decision_vars) -> List[Constraint]:
        """
        Define the constraints for the optimization problem.
        This method should be implemented by the user.

        Args:
            decision_vars (dict): Dictionary of decision variables.

        Returns:
            list: A list of constraints.
        """
        pass

    def define_inverse_problem(self, decision_vars: Dict[str, Variable]):
        """
        Define the inverse problem for the PINN.
        This method should be implemented by the user.

        Args:
            decision_vars (dict): Dictionary of decision variables.
        """
        pass

    def define_optimiser(self, problem: nm.problem.Problem) -> torch.optim.Optimizer:

        lr = self.params.get("learning_rate", 0.003)

        return torch.optim.AdamW(
            problem.parameters(),
            lr=lr,
        )

    def train(self, training_loader: DataLoader, device=torch.device("cpu")):
        try:
            self.logger.info("Defining neural network")
            neural_net = self.define_neural_network()

            input_var_name = self.params.get("input_var", ["p"])
            output_var_name = self.params.get("output_var", ["x"])
            pde_net = nm.system.Node(
                neural_net, input_var_name, output_var_name, name="net"
            )

            self.logger.info(
                f"Symbolic inputs, outputs of the pde_net: {pde_net.input_keys}, {pde_net.output_keys}"
            )

            self.logger.info("Defining decision variables")
            decision_vars = self.define_decision_variables()

            self.logger.info("Defining residual PDE")
            residual_pde = self.define_residual_pde(decision_vars)

            if isinstance(residual_pde, dict):
                for key, value in residual_pde.items():
                    value.show(f"./data/02_intermediate/residual_pde_{key}.png")
            else:
                residual_pde.show("./data/02_intermediate/residual_pde.png")

            self.logger.info("Defining objective function")
            objective_function = self.define_objective_function(
                decision_vars=decision_vars,
                residual_pde=residual_pde,
                training_loader=training_loader,
            )

            self.logger.info("Defining constraints")
            constraints = self.define_constraints(decision_vars)

            self.logger.info("Constructing optimization problem")
            loss_function = nm.loss.PenaltyLoss(
                objectives=objective_function, constraints=constraints
            )
            problem = nm.problem.Problem(
                nodes=[pde_net], loss=loss_function, grad_inference=True
            )

            problem.show("./data/02_intermediate/optimization_problem.png")

            optimizer = self.define_optimiser(problem=problem)

            self.logger.info("Setting up the trainer")

            epochs = self.params.get("epochs", 8000)

            customBasicLogger = CustomBasicLogger(logger=self.logger)
            tqdmCallback = TQDMCallback()

            trainer = Trainer(
                problem=problem.to(device),
                train_data=training_loader,
                optimizer=optimizer,
                epochs=epochs,
                train_metric="train_loss",
                dev_metric="train_loss",
                eval_metric="train_loss",
                warmup=self.params.get("warmup", epochs),
                device=device,
                callback=tqdmCallback,
                logger=customBasicLogger,
            )

            self.logger.info("Training the model")
            best_model = trainer.train()

            self.logger.info("Loading the best state")
            problem.load_state_dict(best_model)

            self.model = problem

            # Inverse problem
            self.define_inverse_problem(decision_vars=decision_vars)

        except Exception as e:
            raise Exception(
                "Error occurred during training. {} error: {}".format(
                    type(e).__name__, str(e)
                )
            )

        # Return the trained model
        return self.model.nodes[0].cpu()


class TQDMCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_progress = None

    def begin_train(self, trainer):
        self.epoch_progress = tqdm(
            total=trainer.epochs,
            desc="Training",
            unit="epoch",
        )

    def end_epoch(self, trainer, output):
        if self.epoch_progress is not None:
            self.epoch_progress.update(1)

    def end_train(self, trainer, output):
        if self.epoch_progress is not None:
            self.epoch_progress.close()


class CustomBasicLogger(BasicLogger):
    def __init__(
        self,
        logger: logging.Logger,
        args=None,
        savedir="./data/06_models",
        verbosity=10,
        stdout=(
            "nstep_dev_loss",
            "loop_dev_loss",
            "best_loop_dev_loss",
            "nstep_dev_ref_loss",
            "loop_dev_ref_loss",
        ),
    ):
        self.logger = logger
        # Initialize the parent class with the required arguments
        super().__init__(args=args, savedir=savedir, verbosity=verbosity, stdout=stdout)

    def log_metrics(self, output, step=None):
        pass

    def log_parameters(self):
        self.logger.info(f"Experiment parameters: {self.args}")

    def log_weights(self, model):
        nweights = sum([i.numel() for i in list(model.parameters()) if i.requires_grad])
        self.logger.info(f"Number of parameters: {nweights}")
        return nweights
