# Evidential Uncertainty Quantification in Physics-Informed Neural Networks (PINNs)

This repository contains the implementation and experimental results for our extended abstract on evidential uncertainty quantification in Physics-Informed Neural Networks (PINNs). We introduce a approach that combines Deep Evidential Regression (DER) with PINNs to provide reliable uncertainty estimates for physics-based predictions.

## 📋 Abstract

We integrate a state-of-the-art method to quantify aleatoric and epistemic uncertainties in physics-informed neural networks and observe that they can be captured effectively while maintaining predictive accuracy.

### Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd pinn-der-ai4x
   ```

2. **Install dependencies using Poetry**:

   ```bash
   poetry install
   ```

## 🚀 Usage

### Quick Start

The repository contains two main experimental setups:

#### 1. Burgers' Equation Experiment

```bash
cd examples/burgers-kedro-experiment
kedro run
```

#### 2. Laplace Equation Experiment

```bash
cd examples/laplace-kedro-experiment
kedro run
```

### Manual Implementation

For custom PDEs, you can use the core library directly:

```python
from pinn_der_ai4x import PINNTrainer, DERMLP, NIG_REG, NIG_NLL

class CustomPINNTrainer(PINNTrainer):
    def define_neural_network(self):
        return DERMLP(
            insize=input_size,
            outsize=output_size,
            hsizes=hidden_sizes,
            nonlin=torch.nn.Tanh,
        )

    def define_objective_function(self, decision_vars, residual_pde):
        # Define your evidential loss functions here
        nig_nll = NIG_NLL(metric, v, alpha, beta, "nig_nll", scaling)
        nig_reg = NIG_REG(metric, v, alpha, beta, "nig_reg", scaling)
        return [nig_nll, nig_reg]
```

## 📊 Experimental Results

### Burgers' Equation

- **Domain**: x ∈ [-1, 1], t ∈ [0, 1]
- **Viscosity**: ν = 0.01/π
- **Network**: 2 hidden layers with 32 neurons each
- **Activation**: Tanh

### Laplace Equation

- **Domain**: 2D rectangular domain
- **Boundary Conditions**: Mixed Dirichlet/Neumann
- **Network**: 2 hidden layers with 30 neurons each
- **Activation**: SiLU

## 🔬 Methodology

### Deep Evidential Regression (DER)

Our implementation uses the Normal Inverse Gamma (NIG) distribution to model:

- **μ**: Mean prediction
- **ν**: Evidence parameter (aleatoric uncertainty)
- **α**: Shape parameter (epistemic uncertainty)
- **β**: Scale parameter

### Loss Functions

1. **NIG_NLL**: Negative log-likelihood loss for evidential learning
2. **NIG_REG**: Regularization term to prevent overconfidence
3. **PDE Residual**: Physics-informed constraint

## 📁 Repository Structure

```text
pinn-der-ai4x/
├── pinn_der_ai4x/           # Core library
│   ├── der_pinn_lib.py      # DER-PINN implementation
│   └── __init__.py
├── examples/
│   ├── burgers-experiment/  # Simple Burgers' equation
│   ├── burgers-kedro-experiment/  # Kedro pipeline for Burgers'
│   ├── laplace-experiment/  # Simple Laplace equation
│   └── laplace-kedro-experiment/  # Kedro pipeline for Laplace
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **Neuromancer**: Optimization and constraint handling
- **NumPy**: Numerical computations
- **Kedro**: Data pipeline management (for experiments)

## 📄 Citation

If you use this code in your research, please cite our extended abstract:

```bibtex
@inproceedings{
   kai2025quantifying,
   title={Quantifying Uncertainty in Physics-Informed Neural Networks},
   author={Yip Jun Kai and Eduardo de Conto and Arvind Easwaran},
   booktitle={AI4X 2025 International Conference},
   year={2025},
   url={https://openreview.net/forum?id=tXJ2G0g9HM}
}
```

## 👨‍💻 Authors & Maintainers

| Name             | Connect                                                                                |
| ---------------- | -------------------------------------------------------------------------------------- |
| Yip Jun Kai      | [LinkedIn](https://www.linkedin.com/in/yipjk/), [GitHub](https://github.com/yipjunkai) |
| Eduardo de Conto |                                                                                        |
| Arvind Easwaran  |                                                                                        |

---

**Note**: This repository accompanies our extended abstract submission to AI4X 2025. For detailed theoretical foundations and experimental results, please refer to the full extended abstract.
