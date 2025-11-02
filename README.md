Vision-Linear-regression

[readme_sgd.md](https://github.com/user-attachments/files/23291485/readme_sgd.md)
# Linear Regression with Stochastic Gradient Descent (SGD)

A from-scratch implementation of Linear Regression using Stochastic Gradient Descent without any built-in machine learning libraries.

## 📋 Overview

This project demonstrates a pure Python implementation of Linear Regression using the SGD optimization algorithm. The implementation uses only NumPy for numerical computations and Matplotlib for visualization, without relying on scikit-learn, TensorFlow, or any other ML framework.

## 🎯 Features

- **Pure Implementation**: Built from scratch without using sklearn or other ML libraries
- **Stochastic Gradient Descent**: Updates parameters using one sample at a time
- **Visualization**: Plots regression line and training loss over time
- **Performance Metrics**: Calculates MSE and R² score
- **Train-Test Split**: Evaluates model on unseen data

## 📁 Project Structure

```
├── linear_regression_sgd.py    # Main implementation file
├── README.md                    # This file
└── regression_results.png       # Generated plot (after running)
```

## 🔧 Requirements

```bash
numpy
matplotlib
```

Install dependencies:
```bash
pip install numpy matplotlib
```

## 🚀 Usage

### Basic Usage

Run the script directly:
```bash
python linear_regression_sgd.py
```

This will:
1. Generate synthetic data with 200 samples
2. Split data into 80% training and 20% testing
3. Train the model using SGD
4. Display performance metrics
5. Save a visualization plot

### Using the Class in Your Code

```python
from linear_regression_sgd import LinearRegressionSGD
import numpy as np

# Prepare your data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train the model
model = LinearRegressionSGD(learning_rate=0.01, n_iterations=100)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate the model
r2_score = model.score(X, y)
print(f"R² Score: {r2_score}")
```

## 🧮 Algorithm Explanation

### Linear Regression Model

The linear regression model predicts output using:

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Where:
- `w` = weights (coefficients)
- `b` = bias (intercept)
- `x` = input features

### Stochastic Gradient Descent

SGD updates parameters by:

1. **Random Sampling**: Select one training example at random
2. **Prediction**: Calculate predicted value using current parameters
3. **Error Calculation**: Compute the difference between prediction and actual value
4. **Gradient Computation**: Calculate gradients of the loss with respect to parameters
5. **Parameter Update**: Update weights and bias in the opposite direction of gradients

**Update Rules:**
```
w = w - α * ∂L/∂w
b = b - α * ∂L/∂b
```

Where:
- `α` = learning rate
- `∂L/∂w` = gradient of loss with respect to weights
- `∂L/∂b` = gradient of loss with respect to bias

### Loss Function

We use Mean Squared Error (MSE):

```
MSE = (1/n) Σ(y_true - y_pred)²
```

## 🎛️ Hyperparameters

- **learning_rate** (default: 0.01): Controls the step size during optimization
  - Too high: May cause divergence
  - Too low: Slow convergence

- **n_iterations** (default: 1000): Number of complete passes through the dataset
  - More iterations generally improve convergence
  - May cause overfitting if too high

- **random_state** (default: 42): Seed for reproducibility

## 📊 Output Example

```
==================================================
MODEL RESULTS
==================================================

Learned Parameters:
  Weights: [4.8523]
  Bias: -2.3456

Training Metrics:
  MSE: 223.4567
  R² Score: 0.9234

Test Metrics:
  MSE: 245.6789
  R² Score: 0.9156
==================================================
```

## 📈 Visualization

The script generates a two-panel plot:

1. **Left Panel**: Scatter plot of data points with the fitted regression line
2. **Right Panel**: Training loss (MSE) over iterations showing convergence

## 🔍 Key Differences from Standard GD

| Feature | Standard GD | Stochastic GD |
|---------|------------|---------------|
| Samples per update | All (batch) | One |
| Update frequency | Once per epoch | n_samples per epoch |
| Convergence | Smooth | Noisy but faster |
| Memory usage | High | Low |
| Best for | Small datasets | Large datasets |

## 🧪 Testing with Custom Data

```python
# Example with multiple features
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([5, 7, 9, 11])

model = LinearRegressionSGD(learning_rate=0.01, n_iterations=500)
model.fit(X, y)

# Get learned parameters
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias}")
```

## 💡 Tips for Better Performance

1. **Feature Scaling**: Normalize features to similar ranges for faster convergence
2. **Learning Rate Tuning**: Start with 0.01 and adjust based on loss behavior
3. **Iteration Count**: Monitor loss history to determine optimal number of iterations
4. **Data Shuffling**: The implementation shuffles data each epoch for better convergence

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📝 License

This project is open source and available for educational purposes.



