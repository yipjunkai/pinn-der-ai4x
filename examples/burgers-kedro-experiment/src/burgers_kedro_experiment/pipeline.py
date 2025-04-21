from kedro.pipeline import Pipeline, node, pipeline
from kedro_umbrella import coder, processor, trainer
from kedro_umbrella.library import *

from .pinn_trainer import DiffusionPINNTrainer
import torch
from neuromancer.dataset import DictDataset
import math


# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            processor(func=load_device, inputs="parameters", outputs="device"),
            processor(
                func=load_mat, inputs="parameters", outputs=["raw_input", "raw_output"]
            ),
            processor(
                func=load_raw_data,
                inputs=["raw_input", "raw_output", "parameters"],
                outputs=["X", "T", "Y"],
                name="load_raw_data",
            ),
            processor(
                func=create_training_dataloaders,
                inputs=["X", "T", "Y", "parameters", "device"],
                outputs=[
                    "train_loader",
                    "test_data",
                    "kwargs",
                ],
                name="create_data",
            ),
            trainer(
                func=checkpoint_training_run,
                name="trainer",
                inputs=[
                    "train_loader",
                    "test_data",
                    "X",
                    "T",
                    "Y",
                    "device",
                    "kwargs",
                    "parameters",
                ],
                outputs="pinn_model_pred",
            ),
            processor(
                func=transform_pred,
                inputs=["pinn_model_pred", "test_data", "parameters"],
                outputs="Y_pred",
            ),
            processor(
                func=score,
                name="score",
                inputs=["Y", "Y_pred"],
                outputs=["nrmse", "r2"],
            ),
        ]
    )


def load_raw_data(
    raw_X: np.ndarray | list[np.ndarray],
    raw_Y: np.ndarray | list[np.ndarray],
    parameters: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, t = raw_X[0], raw_X[1]
    ysol = raw_Y[0]

    X, T = np.meshgrid(x, t)
    X = torch.tensor(X.T).float()
    T = torch.tensor(T.T).float()
    Y = torch.tensor(ysol).float()

    return X, T, Y


def create_training_dataloaders(X, T, Y, parameters, device):
    Nu = parameters.get("Nu", 200)
    Nf = parameters.get("Nf", 1000)
    batch_size = parameters.get("batch_size", None)

    X_test = X.reshape(-1, 1)
    T_test = T.reshape(-1, 1)
    Y_test = Y.reshape(-1, 1)

    left_X = X[:, [0]]
    left_T = T[:, [0]]
    left_Y = -torch.sin(np.pi * left_X[:, 0]).unsqueeze(1)

    bottom_X = X[[0], :].T
    bottom_T = T[[0], :].T
    bottom_Y = torch.zeros(bottom_X.shape[0], 1)

    top_X = X[[-1], :].T
    top_T = T[[-1], :].T
    top_Y = torch.zeros(top_X.shape[0], 1)

    X_train = torch.vstack([left_X, bottom_X, top_X])
    T_train = torch.vstack([left_T, bottom_T, top_T])
    Y_train = torch.vstack([left_Y, bottom_Y, top_Y])

    idx = np.sort(np.random.choice(X_train.shape[0], Nu, replace=False))
    X_train_Nu = X_train[idx, :].float()
    T_train_Nu = T_train[idx, :].float()
    Y_train_Nu = Y_train[idx, :].float()

    x_lb = X_test[0]
    x_ub = X_test[-1]

    t_lb = T_test[0]
    t_ub = T_test[-1]

    X_train_CP = torch.FloatTensor(Nf, 1).uniform_(float(x_lb), float(x_ub))
    T_train_CP = torch.FloatTensor(Nf, 1).uniform_(float(t_lb), float(t_ub))

    X_train_Nf = torch.vstack((X_train_CP, X_train_Nu)).float()
    T_train_Nf = torch.vstack((T_train_CP, T_train_Nu)).float()

    X_train_Nf.requires_grad = True
    T_train_Nf.requires_grad = True

    train_data = DictDataset(
        {
            "x": X_train_Nf,
            "t": T_train_Nf,
        },
        name="train",
    )

    test_data = DictDataset({"x": X_test, "t": T_test, "y": Y_test}, name="test")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=X_train_Nf.shape[0],
        collate_fn=train_data.collate_fn,
        shuffle=False,
    )

    return (
        train_loader,
        test_data,
        {"Y_train_Nu": Y_train_Nu.to(device)},
    )


def checkpoint_training_run(
    train_loader, test_data, X, T, Y, device: torch.device, kwargs, parameters
):

    no_of_loops = math.floor(parameters.get("epochs", 5000) / 10)

    trainer = DiffusionPINNTrainer(params={**parameters, "epochs": 10}, **kwargs)
    model = None

    mse_list, rmse_list, r2_list, max_error_list = [], [], [], []

    for i in range(no_of_loops):

        model = trainer.train(training_loader=train_loader, device=device)

        U_pinn = model(test_data.datadict)["u"]

        u_mean = U_pinn[:, 0].reshape(shape=[256, 100]).detach().cpu()
        u_v = U_pinn[:, 1].reshape(shape=[256, 100]).detach().cpu()
        u_alpha = U_pinn[:, 2].reshape(shape=[256, 100]).detach().cpu()
        u_beta = U_pinn[:, 3].reshape(shape=[256, 100]).detach().cpu()

        _plot_2D3D(
            X=X,
            Y=T,
            true_Z=Y,
            Z=u_mean,
            x_label="x",
            y_label="t",
            z_label="u(x,t)",
            file_name=f"pred_output_{(i + 1) * 10}",
            z_lim=(-1, 1),
            epoch=(i + 1) * 10,
        )

        _plot_uncertainty(
            X=X,
            Y=T,
            Z=u_mean,
            Z_v=u_v,
            Z_alpha=u_alpha,
            Z_beta=u_beta,
            x_label="x",
            y_label="t",
            z_label="u(x,t)",
            file_name=f"uncertainty_output_{(i + 1) * 10}",
            z_lim=(-1, 1),
            epoch=(i + 1) * 10,
        )

        # Training loss
        with open("./data/07_model_output/train_loss.txt", "w+") as f:
            # each line is a loss value
            losses = [float(line.strip()) for line in f.readlines()]

            plot_train_loss(losses, file_name="total_train_loss")

            f.close()

        # Metrics
        mse, rmse, r2, max_error = metrics(Y, u_mean)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        max_error_list.append(max_error)

        graph_metrics(mse_list, rmse_list, r2_list, max_error_list)

    return model


def plot_train_loss(
    train_loss: list[float],
    output_path: str = "./data/07_model_output",
    file_name: str = "train_loss",
    file_type: Literal["png", "jpg", "jpeg"] = "png",
):
    fig = plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid()
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(f"{output_path}/{file_name}.{file_type}")
    logger.info(f"Plot saved at {output_path}/{file_name}.{file_type}")

    plt.close()


def transform_pred(funct, test_data, parameters):
    U_pinn = funct(test_data.datadict)["u"]

    u_mean = U_pinn[:, 0].reshape(shape=[256, 100]).detach().cpu()
    u_v = U_pinn[:, 1].reshape(shape=[256, 100]).detach().cpu()
    u_alpha = U_pinn[:, 2].reshape(shape=[256, 100]).detach().cpu()
    u_beta = U_pinn[:, 3].reshape(shape=[256, 100]).detach().cpu()

    return u_mean


def metrics(Y_test, Y_pred):
    if isinstance(Y_test, torch.Tensor):
        Y_test = Y_test.detach().numpy()

    if isinstance(Y_pred, torch.Tensor):
        Y_pred = Y_pred.detach().numpy()

    mse = mean_squared_error(Y_test, Y_pred)
    rmse = root_mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    max_error = np.max(np.abs(Y_test - Y_pred))

    return mse, rmse, r2, max_error


def graph_metrics(mse, rmse, r2, max_error):
    # plot 4 graphs
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].plot(range(10, 10 * (len(mse) + 1), 10), mse)
    axs[0, 0].set_title("MSE")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("MSE")

    axs[0, 1].plot(range(10, 10 * (len(rmse) + 1), 10), rmse)
    axs[0, 1].set_title("RMSE")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("RMSE")

    axs[1, 0].plot(range(10, 10 * (len(r2) + 1), 10), r2)
    axs[1, 0].set_title("R2")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("R2")

    axs[1, 1].plot(range(10, 10 * (len(max_error) + 1), 10), max_error)
    axs[1, 1].set_title("Max Error")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Max Error")

    # Notate the last value of each metric
    for i, metric in enumerate([mse, rmse, r2, max_error]):
        axs[i // 2, i % 2].text(
            len(metric),
            metric[-1],
            f"{metric[-1]:.4f}",
            ha="center",
            va="center",
            color="red",
        )

    plt.tight_layout()
    plt.savefig("./data/07_model_output/metrics.png")

    plt.close()


def _plot_uncertainty(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    Z_v: np.ndarray | torch.Tensor,
    Z_alpha: np.ndarray | torch.Tensor,
    Z_beta: np.ndarray | torch.Tensor,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "u(x,y)",
    output_path: str = "./data/07_model_output",
    file_name: str = "model_output",
    file_type: Literal["png", "jpg", "jpeg"] = "png",
    x_lim: tuple = None,
    y_lim: tuple = None,
    z_lim: tuple = None,
    epoch: int = None,
):
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().numpy()
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().numpy()
    if isinstance(Z_v, torch.Tensor):
        Z_v = Z_v.detach().numpy()
    if isinstance(Z_alpha, torch.Tensor):
        Z_alpha = Z_alpha.detach().numpy()
    if isinstance(Z_beta, torch.Tensor):
        Z_beta = Z_beta.detach().numpy()

    fig = plt.figure(
        figsize=(16, 8),
    )

    if epoch is not None:
        fig.suptitle(f"Epoch: {epoch}", fontsize=16)

    ale_uncertainty = Z_beta / (Z_alpha - 1)
    eps_uncertainty = Z_beta / (Z_v * (Z_alpha - 1))

    ax1 = fig.add_subplot(121, projection="3d")
    surf_1 = ax1.plot_surface(
        X, Y, Z, facecolors=plt.cm.viridis(ale_uncertainty / ale_uncertainty.max())
    )
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_zlabel(z_label)
    ax1.set_title("Aleatoric Uncertainty")

    ax2 = fig.add_subplot(122, projection="3d")
    surf_2 = ax2.plot_surface(
        X, Y, Z, facecolors=plt.cm.viridis(eps_uncertainty / eps_uncertainty.max())
    )
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_zlabel(z_label)
    ax2.set_title("Epistemic Uncertainty")

    fig.colorbar(surf_1, ax=ax1, pad=0.1, shrink=0.6)
    fig.colorbar(surf_2, ax=ax2, pad=0.1, shrink=0.6)

    if x_lim is not None:
        ax1.set_xlim(x_lim)
        ax2.set_xlim(x_lim)
    if y_lim is not None:
        ax1.set_ylim(y_lim)
        ax2.set_ylim(y_lim)
    if z_lim is not None:
        ax1.set_zlim(z_lim)
        ax2.set_zlim(z_lim)

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{output_path}/{file_name}.{file_type}")
    print(f"Plot saved at {output_path}/{file_name}.{file_type}")

    plt.close()


def _plot_2D3D(
    X: np.ndarray | torch.Tensor,
    Y: np.ndarray | torch.Tensor,
    true_Z: np.ndarray | torch.Tensor,
    Z: np.ndarray | torch.Tensor,
    x_label: str = "x",
    y_label: str = "y",
    z_label: str = "u(x,y)",
    output_path: str = "./data/07_model_output",
    file_name: str = "model_output",
    file_type: Literal["png", "jpg", "jpeg"] = "png",
    x_lim: tuple = None,
    y_lim: tuple = None,
    z_lim: tuple = None,
    epoch: int = None,
):
    # Convert tensors to numpy arrays
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().numpy()
    if isinstance(true_Z, torch.Tensor):
        true_Z = true_Z.detach().numpy()
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().numpy()

    # Create the plot
    fig = plt.figure(
        figsize=(20, 12),
    )

    if epoch is not None:
        fig.suptitle(f"Epoch: {epoch}", fontsize=16)

    for i, (data, title) in enumerate(
        zip([true_Z, Z, true_Z - Z], ["Truth", "Prediction", "Difference"])
    ):
        ax1 = fig.add_subplot(2, 3, i + 1)
        ax1.contourf(X, Y, data, 20, cmap="viridis")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title(title)

        ax1.set_aspect("equal")

        ax2 = fig.add_subplot(2, 3, i + 4, projection="3d")
        ax2.plot_surface(X, Y, data, cmap="viridis")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)
        ax2.set_zlabel(z_label)

        if x_lim is not None:
            ax1.set_xlim(x_lim)
            ax2.set_xlim(x_lim)
        if y_lim is not None:
            ax1.set_ylim(y_lim)
            ax2.set_ylim(y_lim)
        if z_lim is not None:
            ax2.set_zlim(z_lim)

    plt.tight_layout()

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Save the figure
    plt.savefig(f"{output_path}/{file_name}.{file_type}")
    print(f"Plot saved at {output_path}/{file_name}.{file_type}")

    plt.close()
