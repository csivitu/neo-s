import numpy as np

def sigmoid(z):
    try:
        # Ensure z is a numpy array for vectorized operations
        z = np.asarray(z)

        # Check for NaN values
        if np.any(np.isnan(z)):
            raise ValueError("Input contains NaN values.")
        
        # To prevent overflow, we can use np.clip to limit the input range
        z_clipped = np.clip(z, -500, 500)  # Clipping to avoid overflow in exp
        return 1 / (1 + np.exp(-z_clipped))
    
    except OverflowError:
        print("Overflow error in sigmoid calculation.")
        return np.where(z > 0, 1, 0.0)  # Return 1 for large positive z, 0 for large negative z
    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Return None or handle the error as needed

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=50, batch_size=4, regularization_strength=0.01, use_regularization=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization_strength = regularization_strength
        self.use_regularization = use_regularization

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.ones(n_features)  # Proper weight initialization
        self.bias = 0.0  # Bias should be a scalar, not an array

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_predicted = sigmoid(linear_model)

                # Handle case where sigmoid returns None (error in calculation)
                if y_predicted is None:
                    print("Error during prediction, skipping this batch.")
                    continue

                dw = (1 / len(X_batch)) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1 / len(X_batch)) * np.sum(y_predicted - y_batch)

                if self.use_regularization:
                    dw += (self.regularization_strength / len(X_batch)) * self.weights

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db  # Correctly updates the scalar bias

            # Improved stopping condition based on weight updates
            if np.linalg.norm(dw) < 0.001:
                print(f"Stopping early at epoch {epoch}")
                break

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        
        # Handle potential error from sigmoid function
        if y_predicted is None:  
            print("Error during prediction.")
            return None
            
        # Clear threshold for prediction
        y_class_pred = (y_predicted >= 0.5).astype(int)  
        return y_class_pred

# Sample training data
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1])

# Model instantiation and training
model = LogisticRegression(learning_rate=0.0001, epochs=5000, batch_size=2, regularization_strength=0.5)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_train)
print("Predicted classes:", predictions)
