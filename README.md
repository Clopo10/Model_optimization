# Model Optimization for CIFAR-10

This project documents a CIFAR-10 image classification workflow focused on improving model quality under modest GPU constraints. It includes the training notebook, hyperparameter tuning runs, a final optimized Keras model, and a written performance review.

## Project Overview

The work in this repository follows four main stages:

1. Build and validate a baseline CNN that trains reliably on the target hardware.
2. Explore hyperparameters with grid search, random search, and Bayesian optimization.
3. Compare the best tuned architecture against a longer training run to study overfitting.
4. Produce a final optimized model with early stopping and save it as a reusable artifact.

The project was developed for CIFAR-10 image classification and tuned with hardware safety in mind for an Nvidia GTX 1050 Ti 4GB GPU.

## Repository Contents

- [models/image.ipynb](models/image.ipynb): main notebook for preprocessing, training, tuning, and evaluation.
- [models/cifar_tuning/](models/cifar_tuning/): saved tuner outputs for Bayesian, grid, and random searches.
- [deliverables/optimized_cifar10_model.keras](deliverables/optimized_cifar10_model.keras): exported final model.
- [deliverables/Performance_improvement.md](deliverables/Performance_improvement.md): summary of the training and tuning results.

## Results Summary

The performance review documents the key outcomes of the project:

- Baseline CNN after 3 epochs: 61.08% validation accuracy.
- Grid search best result: 68.65% validation accuracy.
- Random search best result: 69.30% validation accuracy.
- Bayesian optimization best result: 72.23% validation accuracy.
- Final model training used early stopping to preserve the best weights and avoid extended overfitting.

## Requirements

The project uses Python with TensorFlow, Keras Tuner, scikit-learn, NumPy, Matplotlib, and related notebook tooling. A full dependency list is available in [requirements.txt](requirements.txt).

## Getting Started

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open [models/image.ipynb](models/image.ipynb) in Jupyter or VS Code.
4. Run the notebook cells in order to reproduce the pipeline, tuning runs, and evaluation plots.

## Notes

- The saved tuning directories under `models/cifar_tuning/` contain the trial metadata and checkpoints produced during search.
- The final `.keras` file in `deliverables/` can be loaded directly with Keras for inference or further experimentation.
- The notebook and report are the best references if you want the exact experimental setup and interpretation of the results.

## Reusing the Model

Example loading snippet:

```python
from keras.models import load_model

model = load_model("deliverables/optimized_cifar10_model.keras")
```

You can then evaluate `model` on CIFAR-10 inputs or use it as the starting point for additional fine-tuning.
