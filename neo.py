import numpy as np

def sigmoid(z):
branch4
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

=======
    try:
        return 1 / (1 + np.exp(-z))
    except OverflowError as e:
issue_2_branch
=======
        print(f"OverflowError in sigmoid: {e}")
main
        return 1.0 if z > 0 else 0.0
    
main
class LogisticRegression:
issue_3_branch
    def compute_loss(self, X, y):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        log_loss = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        
        if self.use_regularization:
            log_loss += (self.regularization_strength / 2) * np.sum(np.square(self.weights))  # L2 regularization
        
        return log_loss

    def __init__(self, learning_rate=0.01, epochs=50, batch_size=4, regularization_strength=0.01, use_regularization=True):
=======
    def __init__(self, learning_rate=0.01, epochs=50, batch_size=4, regularization_strength=0.01, use_regularization=True, learning_rate_deacy = 0.99):
main
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization_strength = regularization_strength
        self.use_regularization = use_regularization
        self.learning_rate_decay = learning_rate_deacy

    def fit(self, X, y):
issue_2_branch
        try:
            n_samples, n_features = X.shape
            self.weights = np.zeros(n_features)  # Corrected weight initialization
            self.bias = 0  # Corrected bias initialization

            prev_weights = np.zeros(n_features)

            for epoch in range(self.epochs):
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                for i in range(0, n_samples, self.batch_size):
                    X_batch = X_shuffled[i:i + self.batch_size]
                    y_batch = y_shuffled[i:i + self.batch_size]
=======
        n_samples, n_features = X.shape
main
        self.weights = np.random.rand(n_features) * 0.01  # Error 1: Improper weight initialization
        self.bias = 0  # Error 2: Bias should be a scalar, not an array
=======
branch4
        self.weights = np.zeros(n_features)  # Initialize weights to zeros for better convergence
        self.bias = 0.0  # Bias should be a scalar
=======
        self.weights = np.random.randn(n_features)  # Corrected weight initialization
        self.bias = 0  # Corrected bias initialization

        prev_weights = np.zeros(n_features)
        prev_bias = 0
main
main
main


                    linear_model = np.dot(X_batch, self.weights) + self.bias
                    y_predicted = sigmoid(linear_model)

                    dw = (1 / len(X_batch)) * np.dot(X_batch.T, (y_predicted - y_batch))
                    db = (1 / len(X_batch)) * np.sum(y_predicted - y_batch)

branch4
                # Calculate gradients
                dw = (1 / len(X_batch)) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1 / len(X_batch)) * np.sum(y_predicted - y_batch)

                # Apply regularization if required
                if self.use_regularization:
main
                    dw += (self.regularization_strength / n_samples) * self.weights  # Error 3: Regularization applied incorrectly
=======
                    dw += (self.regularization_strength / len(X_batch)) * self.weights
main

                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

main
            if np.linalg.norm(dw)  and np.linalg.norm(db)< 0.001:
                break  # Error 5: Inadequate stopping condition
=======
            # Improved stopping condition
            weight_update_norm = np.linalg.norm(dw)
            if weight_update_norm < 0.001:
                print(f"Stopping early at epoch {epoch} with weight update norm: {weight_update_norm:.6f}")
                break
main

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
main
        y_class_pred = (y_predicted > 0.5).astype(int)  
        ambiguous_indices = np.where(y_predicted == 0.5)[0]
        
        if ambiguous_indices.size > 0:
            random_choices = np.random.choice([0, 1], size=ambiguous_indices.size)
            y_class_pred[ambiguous_indices] = random_choices
    
=======
        y_class_pred = (y_predicted >= 0.5).astype(int)  # More concise and clear prediction
main
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
=======
                    if self.use_regularization:
                        dw += (self.regularization_strength * self.weights)  # Corrected regularization term

issue_2_branch
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db  # Corrected bias update logic
=======
                if self.use_regularization:
                    dw += (self.regularization_strength * self.weights)  # Corrected regularization term
                    dw += (self.regularization_strength * self.bias)
main

                if np.allclose(prev_weights, self.weights, rtol=1e-05):  # Corrected stopping condition
                    break

issue_2_branch
                prev_weights = self.weights
        
        except ValueError as e:
            print(f"ValueError in fit method: {e}")
        
        except TypeError as e:
            print(f"TypeError in fit method: {e}")

        except IndexError as e:
            print(f"IndexError in fit method: {e}")

        except Exception as e:
            print(f"Unexpected error in fit method: {e}")        
=======
            self.learning_rate *= self.learning_rate_decay

            loss = self.compute_loss(X, y)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}')            
            
            if np.allclose(prev_weights, self.weights, rtol=1e-05):  # Corrected stopping condition
                break

            prev_weights = np.copy(self.weights)
            prev_bias = self.bias

        print(f"Epoch {epoch}: Weights change: {np.linalg.norm(dw)}, Bias change: {abs(db)}")    

main

    def predict(self, X):
        try:
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)
            y_class_pred = [1 if i > 0.5 else 0 for i in y_predicted]  # Corrected equality condition
            return np.array(y_class_pred)
        
        except ValueError as e:
            print(f"ValueError in fit method: {e}")
        
        except TypeError as e:
            print(f"TypeError in fit method: {e}")

        except Exception as e:
            print(f"Unexpected error in fit method: {e}") 
main
