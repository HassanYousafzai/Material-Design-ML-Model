# Material-Design-ML-Model
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
A hybrid machine learning-simulation model for predicting material mechanical strength, developed by Hassan Mahmood Yousafzai.

## Overview
This project implements a neural network-based model combined with a polynomial residual correction to predict the mechanical strength of materials based on density and thickness. The model leverages simulated data and achieves a Mean Absolute Error (MAE) of 8.99 MPa and a Pearson correlation of 0.997, demonstrating high accuracy for material design applications.

### Key Features
- Neural network architecture with 256-128-64-32-1 layers, incorporating BatchNormalization and Dropout (0.2) for regularization.
- Log-transformation of target values to improve prediction stability.
- Polynomial (degree=2) residual correction scaled by 0.05x to address underprediction.
- Trained using a custom loss function (MSE + 1.26 * MAE) over 700 epochs with the Adam optimizer.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HassanYousafzai/Material-Design-ML-Model.git
   cd Material-Design-ML-Model
   ```
2. Install dependencies:
   ```bash
   pip install torch numpy matplotlib scipy scikit-learn joblib
   ```
3. Ensure you have Git and Python 3.7+ installed.

## Usage
Run the main script to train the model and generate predictions:
```bash
python material_design_model_v48.py
```
- The script trains the model, saves weights (`material_model.pth`) and correction model (`residual_correction_model.pkl`), and displays a loss plot.
- Example prediction for density=2.5, thickness=0.5 yields ~647.11 MPa (offset 4.97 MPa from simulated 642.14 MPa).
- **Note**: If `material_model.pth` or `residual_correction_model.pkl` are excluded, run the script to regenerate them.

## Files
- `material_design_model_v48.py`: Main Python script containing the model, training loop, and prediction function.
- `material_model.pth`: Saved neural network weights (regenerate if excluded).
- `residual_correction_model.pkl`: Saved residual correction model (regenerate if excluded).
- `loss_plot.png`: Visualization of training and validation loss (generated during runtime).

## Performance
- **MAE**: 8.99 MPa (final epoch).
- **Correlation**: 0.997 (final epoch).
- **Example Prediction**: 647.11 MPa for density=2.5, thickness=0.5 (offset 4.97 MPa from 642.14 MPa).

## Development
- **Author**: Hassan Mahmood Yousafzai
- **Date**: July 2025
- **Tools**: Python, PyTorch, NumPy, Matplotlib, SciPy, scikit-learn
- **License**: MIT License (see LICENSE file)

## Contributing
Feel free to fork this repository, submit issues, or propose enhancements. Contributions are welcome!

## License
This project is licensed under the [MIT License](LICENSE), allowing free use, modification, and distribution, subject to the terms outlined in the LICENSE file.
