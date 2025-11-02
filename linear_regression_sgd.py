import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionSGD:
    """
    Linear Regression implementation using Stochastic Gradient Descent.
    
    This implementation doesn't use any built-in ML libraries like sklearn.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        """
        Initialize the Linear Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            The step size for gradient descent updates
        n_iterations : int
            Number of passes through the dataset
        random_state : int
            Seed for random number generation
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def fit(self, X, y):
        """
        Fit the linear regression model using SGD.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Training data
        y : numpy array of shape (n_samples,)
            Target values
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # SGD optimization
        for iteration in range(self.n_iterations):
            # Shuffle the data for each epoch
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                # Select one sample
                xi = X[idx:idx+1]
                yi = y[idx:idx+1]
                
                # Compute prediction
                y_pred = np.dot(xi, self.weights) + self.bias
                
                # Compute gradients
                dw = -2 * xi.T.dot(yi - y_pred) / 1  # derivative w.r.t. weights
                db = -2 * np.sum(yi - y_pred) / 1     # derivative w.r.t. bias
                
                # Update parameters
                self.weights -= self.learning_rate * dw.flatten()
                self.bias -= self.learning_rate * db
            
            # Calculate loss for the entire dataset (for tracking)
            if iteration % 10 == 0:
                y_pred_all = self.predict(X)
                loss = self._mse(y, y_pred_all)
                self.loss_history.append(loss)
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : numpy array of shape (n_samples,)
            Predicted values
        """
        return np.dot(X, self.weights) + self.bias
    
    def _mse(self, y_true, y_pred):
        """
        Calculate Mean Squared Error.
        
        Parameters:
        -----------
        y_true : numpy array
            True values
        y_pred : numpy array
            Predicted values
            
        Returns:
        --------
        mse : float
            Mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def score(self, X, y):
        """
        Calculate R-squared score.
        
        Parameters:
        -----------
        X : numpy array
            Input data
        y : numpy array
            True values
            
        Returns:
        --------
        r2_score : float
            R-squared score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


def generate_sample_data(n_samples=100, n_features=1, noise=10, random_state=42):
    """
    Generate sample data for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    noise : float
        Standard deviation of noise
    random_state : int
        Random seed
        
    Returns:
    --------
    X, y : numpy arrays
        Generated data and target
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features) * 10
    true_weights = np.random.randn(n_features) * 5
    true_bias = np.random.randn() * 10
    y = np.dot(X, true_weights) + true_bias + np.random.randn(n_samples) * noise
    return X, y


def plot_results(X, y, model):
    """
    Plot the regression line and data points.
    
    Parameters:
    -----------
    X : numpy array
        Input data
    y : numpy array
        Target values
    model : LinearRegressionSGD
        Trained model
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='Data points')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression with SGD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss history
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history)
    plt.xlabel('Iteration (×10)')
    plt.ylabel('Mean Squared Error')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=200, n_features=1, noise=15)
    
    # Split data into train and test sets (80-20 split)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train the model
    print("\nTraining Linear Regression model with SGD...")
    model = LinearRegressionSGD(learning_rate=0.01, n_iterations=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = model._mse(y_train, y_train_pred)
    test_mse = model._mse(y_test, y_test_pred)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL RESULTS")
    print("="*50)
    print(f"\nLearned Parameters:")
    print(f"  Weights: {model.weights}")
    print(f"  Bias: {model.bias:.4f}")
    print(f"\nTraining Metrics:")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"\nTest Metrics:")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  R² Score: {test_r2:.4f}")
    print("="*50)
    
    # Plot results
    plot_results(X_train, y_train, model)
    print("\nPlot saved as 'regression_results.png'")