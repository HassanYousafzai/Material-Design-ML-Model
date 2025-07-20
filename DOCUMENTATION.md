# Documentation: Hybrid ML-Simulation Model for Material Design

## Project Background
This project was developed as part of a research project to optimize material design through machine learning techniques. The model predicts the mechanical strength of materials based on input parameters such as density and thickness, utilizing a hybrid approach that combines a neural network with a residual correction mechanism. The work addresses underprediction challenges and provides a robust tool for computational material science.

## Methodology

### Data Generation
- **Sample Size**: 1000 synthetic data points.
- **Input Features**:
  - Density: Uniformly distributed between 1.0 and 3.0.
  - Thickness: Uniformly distributed between 0.1 and 1.0.
- **Target Variable**:
  - Mechanical strength calculated as \( 500 \times \text{density}^{0.5} \times \text{thickness}^{0.3} + \mathcal{N}(0, 10) \), where \(\mathcal{N}(0, 10)\) represents Gaussian noise with mean 0 and standard deviation 10.
  - Log-transformed to enhance model stability and prediction accuracy.
- **Data Split**: 80% training (800 samples), 20% validation (200 samples), with random selection using `np.random.choice`.

### Model Architecture
- **Neural Network**:
  - **Structure**: 5-layer feedforward network.
    - Input layer: 2 neurons (density, thickness).
    - Hidden layers: 256 → 128 → 64 → 32 neurons, each with ReLU activation, BatchNormalization, and Dropout (0.2) for regularization.
    - Output layer: 1 neuron (log-transformed strength).
  - **Implementation**: Built using PyTorch’s `nn.Module`.
- **Correction Model**:
  - Polynomial regression (degree=2) fitted to residuals (difference between predicted and actual values).
  - Residuals scaled by 0.05x to fine-tune corrections and avoid overcompensation.

### Training
- **Optimizer**: Adam with learning rate 0.0003 and weight decay 1e-5 to prevent overfitting.
- **Loss Function**: Custom loss combining Mean Squared Error (MSE) and Mean Absolute Error (MAE), defined as:
  \[
  \text{Loss} = \text{MSE} + 1.26 \times \text{MAE}
  \]
  The factor 1.26 weights absolute errors to balance the loss contribution.
- **Epochs**: 700.
- **Metrics**: MAE and Pearson correlation computed every 50 epochs on validation data, with predictions exponentiated back to the original scale for evaluation.

## Results
- **Training Loss**: Decreased from 0.4402 to 0.2222 over 700 epochs.
- **Validation Loss**: Decreased from 0.1417 to 0.1006 over 700 epochs.
- **Mean Absolute Error (MAE)**: Stabilized at 8.99 MPa, with a minimum of 8.69 MPa at epoch 200.
- **Pearson Correlation**: Consistently 0.997 from epoch 200 onward, indicating a strong linear relationship between predicted and actual values.
- **Example Prediction**: For density=2.5 and thickness=0.5, predicted strength is 647.11 MPa (offset 4.97 MPa from the simulated value of 642.14 MPa).

## Future Work
- **Data Expansion**: Incorporate real-world experimental data to validate and extend the model.
- **Feature Enhancement**: Add parameters such as temperature, material composition, or stress conditions.
- **Hyperparameter Tuning**: Explore variations in learning rate, layer sizes, or dropout rates to further optimize performance.
- **Deployment**: Develop a lightweight version for integration into material design workflows.

## Author
Hassan Mahmood Yousafzai, July 2025.
