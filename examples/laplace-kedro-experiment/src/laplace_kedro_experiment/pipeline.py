import math
from kedro.pipeline import Pipeline, node, pipeline
from kedro_umbrella import coder, processor, trainer
from kedro_umbrella.library import *

from .pinn_trainer import DiffusionPINNTrainer
import torch
from neuromancer.dataset import DictDataset


# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            processor(func=load_device, inputs="parameters", outputs="device"),
            processor(
                func=gen_mat,
                inputs="parameters",
                outputs=["X", "Y", "T_true"],
                name="generate_input_data",
            ),
            processor(
                func=create_training_dataloaders,
                inputs=["X", "Y", "parameters", "device"],
                outputs=[
                    "train_loader",
                    "test_data",
                    "kwargs",
                ],
                name="create_data_loader",
            ),
            trainer(
                func=checkpoint_training_run,
                name="trainer",
                inputs=[
                    "train_loader",
                    "test_data",
                    "T_true",
                    "device",
                    "kwargs",
                    "parameters",
                ],
                outputs="pinn_model_pred",
            ),
            processor(
                func=transform_pred,
                inputs=["pinn_model_pred", "test_data", "parameters"],
                outputs="T_pred",
            ),
            processor(
                func=score,
                name="score",
                inputs=["T_true", "T_pred"],
                outputs=["nrmse", "r2"],
            ),
        ]
    )


def gen_mat(
    parameters: Dict[str, Any]
) -> Tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
    # Define number of collocation points in x and y
    Nx = parameters.get("Nx", 200)
    Ny = parameters.get("Ny", 200)

    # Define temperature at upper boundary
    To = 1.0

    # Create collocation points
    x = torch.linspace(0, 1, Nx)
    y = torch.linspace(0, 1, Ny)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Compute the analytical solution
    T_exact = np.zeros(shape=(Nx, Ny))
    N_modes = 200
    To = 1
    x_ = np.linspace(0, 1, Nx)
    y_ = np.linspace(0, 1, Ny)

    for k in range(1, N_modes):
        for i in range(Nx):
            for j in range(Ny):
                T_exact[i, j] += (
                    2
                    * To
                    * (1 - (-1) ** k)
                    * np.sin(k * np.pi * x_[i])
                    * np.sinh(k * np.pi * y_[j])
                    / (k * np.pi * np.sinh(k * np.pi))
                )

    return X, Y, T_exact


def create_training_dataloaders(X, Y, parameters, device):

    N_cp = parameters.get("N_cp", 500)  # Number of collocation points

    X_test = X.reshape(-1, 1)
    Y_test = Y.reshape(-1, 1)

    # Initializing the fields
    ic_X = X[:, :]
    ic_Y = Y[:, :]
    ic_T = X[:, :] * 0.0
    ic_T[:, -1] = 1.0

    # Left boundary conditions
    left_bc_X = X[[0], :]
    left_bc_Y = Y[[0], :]
    left_bc_T = ic_T[[0], :]

    # Right boundary conditions
    right_bc_X = X[[-1], :]
    right_bc_Y = Y[[-1], :]
    right_bc_T = ic_T[[-1], :]

    # Top boundary conditions
    top_bc_X = X[:, [-1]]
    top_bc_Y = Y[:, [-1]]
    top_bc_T = ic_T[:, [-1]]

    # Bottom boundary conditions
    bottom_bc_X = X[:, [0]]
    bottom_bc_Y = Y[:, [0]]
    bottom_bc_T = ic_T[:, [0]]

    X_train_bc = torch.concat(
        [
            left_bc_X.flatten(),
            right_bc_X.flatten(),
            top_bc_X.flatten(),
            bottom_bc_X.flatten(),
        ]
    ).view((-1, 1))
    Y_train_bc = torch.concat(
        [
            left_bc_Y.flatten(),
            right_bc_Y.flatten(),
            top_bc_Y.flatten(),
            bottom_bc_Y.flatten(),
        ]
    ).view((-1, 1))
    T_train_bc = torch.concat(
        [
            left_bc_T.flatten(),
            right_bc_T.flatten(),
            top_bc_T.flatten(),
            bottom_bc_T.flatten(),
        ]
    ).view((-1, 1))

    N_bc = X_train_bc.shape[0]

    # Domain bounds
    x_lb = X_test[0]
    x_ub = X_test[-1]
    y_lb = Y_test[0]
    y_ub = Y_test[-1]

    X_train_cp = torch.FloatTensor(N_cp, 1).uniform_(float(x_lb), float(x_ub))
    Y_train_cp = torch.FloatTensor(N_cp, 1).uniform_(float(y_lb), float(y_ub))

    X_train = torch.vstack((X_train_cp, X_train_bc)).float()
    Y_train = torch.vstack((Y_train_cp, Y_train_bc)).float()

    # turn on gradients for PINN
    X_train.requires_grad = True
    Y_train.requires_grad = True

    train_data = DictDataset(
        {
            "x": X_train,
            "y": Y_train,
        },
        name="train",
    )

    test_data = DictDataset({"x": X_test, "y": Y_test}, name="test")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=X_train.shape[0],
        collate_fn=train_data.collate_fn,
        shuffle=False,
    )

    return (
        train_loader,
        test_data,
        {
            "N_bc": N_bc,
            "T_train_bc": T_train_bc.to(device),
            "ic_X": ic_X.to(device),
            "ic_Y": ic_Y.to(device),
        },
    )


def checkpoint_training_run(
    train_loader, test_data, T_true, device, kwargs, parameters
):
    no_of_loops = math.floor(parameters.get("epochs", 5000) / 10)

    trainer = DiffusionPINNTrainer(
        params={
            **parameters,
            "epochs": 10,
        },
        **kwargs,
    )

    model = None

    mse_list, rmse_list, r2_list, max_error_list = [], [], [], []

    for i in range(no_of_loops):
        model = trainer.train(training_loader=train_loader, device=device)

        T_pinn = model(test_data.datadict)["T"]

        t_mean = T_pinn[:, [0]].reshape(shape=[200, 200]).detach().cpu()
        t_v = T_pinn[:, [1]].reshape(shape=[200, 200]).detach().cpu()
        t_alpha = T_pinn[:, [2]].reshape(shape=[200, 200]).detach().cpu()
        t_beta = T_pinn[:, [3]].reshape(shape=[200, 200]).detach().cpu()

        plot_heatmap(
            (i + 1) * 10,
            T_true,
            t_mean,
            kwargs,
            file_name=f"model_output_diff_{(i + 1) * 10:05d}",
        )

        plot_temperature_profile(
            (i + 1) * 10,
            T_true,
            t_mean,
            t_v,
            t_alpha,
            t_beta,
            parameters,
            kwargs,
            file_name=f"temperature_profile_{(i + 1) * 10:05d}",
        )

        # Training loss
        with open("./data/07_model_output/train_loss.txt", "w+") as f:
            # each line is a loss value
            losses = [float(line.strip()) for line in f.readlines()]

            plot_train_loss(losses, file_name="total_train_loss")

            f.close()

        mse, rmse, r2, max_error = metrics(T_true, t_mean)

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
    T_pinn = funct(test_data.datadict)["T"]

    t_mean = T_pinn[:, [0]].reshape(shape=[200, 200]).detach().cpu()
    t_v = T_pinn[:, [1]].reshape(shape=[200, 200]).detach().cpu()
    t_alpha = T_pinn[:, [2]].reshape(shape=[200, 200]).detach().cpu()
    t_beta = T_pinn[:, [3]].reshape(shape=[200, 200]).detach().cpu()

    return t_mean


def metrics(T_true, T_pred):
    if isinstance(T_true, torch.Tensor):
        T_true = T_true.detach().cpu().numpy()
    if isinstance(T_pred, torch.Tensor):
        T_pred = T_pred.detach().cpu().numpy()

    mse = mean_squared_error(T_true, T_pred)
    rmse = root_mean_squared_error(T_true, T_pred)
    r2 = r2_score(T_true, T_pred)
    max_error = np.max(np.abs(T_true - T_pred))

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


def plot_heatmap(epoch, T_exact, T_pinn, kwargs, file_name="model_output_diff"):

    ic_X, ic_Y = kwargs["ic_X"], kwargs["ic_Y"]

    fig = plt.figure(figsize=(20, 6))

    # subtitle epoch count
    fig.suptitle(f"Epoch: {epoch}", fontsize=16, ha="left")

    # Plot for the first heatmap (Exact solution)
    ax1 = plt.subplot(1, 3, 1)
    cbarticks = np.arange(-0.1, 1.1, 0.05)
    CP1 = plt.contourf(ic_X.cpu(), ic_Y.cpu(), T_exact, cbarticks, cmap="viridis")
    plt.title("Exact solution, $T_{exact}$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.colorbar(CP1, label="$T$")
    ax1.set_aspect("equal", adjustable="box")

    # Plot for the second heatmap (PINN solution)
    ax2 = plt.subplot(1, 3, 2)
    CP2 = plt.contourf(ic_X.cpu(), ic_Y.cpu(), T_pinn, cbarticks, cmap="viridis")
    plt.title("PINN solution, $T_{PINN}$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.colorbar(CP2, label="$T$")
    ax2.set_aspect("equal", adjustable="box")

    # Plot for the third heatmap (Absolute error between PINN approximation and exact solution)
    ax3 = plt.subplot(1, 3, 3)
    CP3 = plt.contourf(
        ic_X.cpu(),
        ic_Y.cpu(),
        torch.abs(torch.Tensor(T_exact) - T_pinn),
        cbarticks,
        cmap="viridis",
    )
    plt.title("Absolute error, $|T_{PINN} - T_{exact}|$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.colorbar(CP3, label="$T$")
    ax3.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    output_path: str = "./data/07_model_output"
    file_type: Literal["png", "jpg", "jpeg"] = "png"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(f"{output_path}/{file_name}.{file_type}")
    logger.info(f"Plot saved at {output_path}/{file_name}.{file_type}")

    plt.close()


def plot_temperature_profile(
    epoch,
    T_exact,
    t_mean,
    t_v,
    t_alpha,
    t_beta,
    parameters,
    kwargs,
    file_name="temperature_profile",
):

    if isinstance(T_exact, torch.Tensor):
        T_exact = T_exact.detach().cpu().numpy()
    if isinstance(t_mean, torch.Tensor):
        t_mean = t_mean.detach().cpu().numpy()
    if isinstance(t_v, torch.Tensor):
        t_v = t_v.detach().cpu().numpy()
    if isinstance(t_alpha, torch.Tensor):
        t_alpha = t_alpha.detach().cpu().numpy()
    if isinstance(t_beta, torch.Tensor):
        t_beta = t_beta.detach().cpu().numpy()

    ale_uncertainty = t_beta / (t_alpha - 1)
    eps_uncertainty = t_beta / (t_v * (t_alpha - 1))

    Nx = parameters.get("Nx", 200)
    Ny = parameters.get("Ny", 200)

    x_ = np.linspace(0, 1, Nx)
    y_ = np.linspace(0, 1, Ny)

    # Calculate the horizontal centerline (middle row) for both datasets
    horizontal_centerline_T_exact = T_exact[T_exact.shape[0] // 2, :]
    horizontal_centerline_T_pinn = t_mean[t_mean.shape[0] // 2, :]

    # Calculate the vertical centerline (middle column) for both datasets
    vertical_centerline_T_exact = T_exact[:, T_exact.shape[1] // 2]
    vertical_centerline_T_pinn = t_mean[:, t_mean.shape[1] // 2]

    # Create a figure and axis for the plots
    fig, ax = plt.subplots(2, 2, figsize=(14, 14))

    fig.suptitle(f"Epoch: {epoch}", fontsize=16)

    horizontal_xlim = (-0.1, 1.1)
    horizontal_ylim = (-0.1, 1.3)

    vertical_xlim = (-0.1, 0.3)
    vertical_ylim = (-0.1, 1.1)

    # Plot 1: Temperature at the horizontal centerline vs. x values
    ax[0, 0].plot(x_, horizontal_centerline_T_exact, color="black", label="$T_{exact}$")
    ax[0, 0].plot(x_, horizontal_centerline_T_pinn, color="blue", label="$T_{PINN}$")
    ax[0, 0].fill_between(
        x_,
        horizontal_centerline_T_pinn - ale_uncertainty[T_exact.shape[0] // 2, :],
        horizontal_centerline_T_pinn + ale_uncertainty[T_exact.shape[0] // 2, :],
        color="blue",
        alpha=0.2,
        label="ALE Uncertainty",
    )
    ax[0, 0].set_title("Temperature Profile at Horizontal Centerline")
    ax[0, 0].set_xlabel("$x$")
    ax[0, 0].set_ylabel("$T(x,y=0.5)$")
    ax[0, 0].grid(True, linestyle=":", color="gray")  # Add dotted grid
    ax[0, 0].set_xlim(horizontal_xlim)
    ax[0, 0].set_ylim(horizontal_ylim)
    ax[0, 0].legend()

    # Plot 2: Temperature at the vertical centerline vs. y values
    ax[0, 1].plot(vertical_centerline_T_exact, y_, color="black", label="$T_{exact}$")
    ax[0, 1].plot(vertical_centerline_T_pinn, y_, color="blue", label="$T_{PINN}$")
    ax[0, 1].fill_betweenx(
        y_,
        vertical_centerline_T_pinn - ale_uncertainty[:, T_exact.shape[1] // 2],
        vertical_centerline_T_pinn + ale_uncertainty[:, T_exact.shape[1] // 2],
        color="blue",
        alpha=0.2,
        label="ALE Uncertainty",
    )
    ax[0, 1].set_title("Temperature Profile at Vertical Centerline")
    ax[0, 1].set_xlabel("$T(x=0.5, y)$")
    ax[0, 1].set_ylabel("$y$")
    ax[0, 1].grid(True, linestyle=":", color="gray")  # Add dotted grid
    ax[0, 1].set_xlim(vertical_xlim)
    ax[0, 1].set_ylim(vertical_ylim)
    ax[0, 1].legend()

    # Plot 3: Temperature at the horizontal centerline vs. x values with eps uncertainty
    ax[1, 0].plot(x_, horizontal_centerline_T_exact, color="black", label="$T_{exact}$")
    ax[1, 0].plot(x_, horizontal_centerline_T_pinn, color="blue", label="$T_{PINN}$")
    ax[1, 0].fill_between(
        x_,
        horizontal_centerline_T_pinn - eps_uncertainty[T_exact.shape[0] // 2, :],
        horizontal_centerline_T_pinn + eps_uncertainty[T_exact.shape[0] // 2, :],
        color="blue",
        alpha=0.2,
        label="EPS Uncertainty",
    )
    ax[1, 0].set_title("Temperature Profile at Horizontal Centerline")
    ax[1, 0].set_xlabel("$x$")
    ax[1, 0].set_ylabel("$T(x,y=0.5)$")
    ax[1, 0].grid(True, linestyle=":", color="gray")  # Add dotted grid
    ax[1, 0].set_xlim(horizontal_xlim)
    ax[1, 0].set_ylim(horizontal_ylim)
    ax[1, 0].legend()

    # Plot 4: Temperature at the vertical centerline vs. y values with eps uncertainty
    ax[1, 1].plot(vertical_centerline_T_exact, y_, color="black", label="$T_{exact}$")
    ax[1, 1].plot(vertical_centerline_T_pinn, y_, color="blue", label="$T_{PINN}$")
    ax[1, 1].fill_betweenx(
        y_,
        vertical_centerline_T_pinn - eps_uncertainty[:, T_exact.shape[1] // 2],
        vertical_centerline_T_pinn + eps_uncertainty[:, T_exact.shape[1] // 2],
        color="blue",
        alpha=0.2,
        label="EPS Uncertainty",
    )
    ax[1, 1].set_title("Temperature Profile at Vertical Centerline")
    ax[1, 1].set_xlabel("$T(x=0.5, y)$")
    ax[1, 1].set_ylabel("$y$")
    ax[1, 1].grid(True, linestyle=":", color="gray")  # Add dotted grid
    ax[1, 1].set_xlim(vertical_xlim)
    ax[1, 1].set_ylim(vertical_ylim)
    ax[1, 1].legend()

    plt.tight_layout()

    output_path: str = "./data/07_model_output"
    file_type: Literal["png", "jpg", "jpeg"] = "png"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(f"{output_path}/{file_name}.{file_type}")
    logger.info(f"Plot saved at {output_path}/{file_name}.{file_type}")

    plt.close()
