import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def compute_loss(self, X, y):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        log_loss = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        
        if self.use_regularization:
            log_loss += (self.regularization_strength / 2) * np.sum(np.square(self.weights))  # L2 regularization
        
        return log_loss

    def __init__(self, learning_rate=0.01, epochs=50, batch_size=4, regularization_strength=0.01, use_regularization=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization_strength = regularization_strength
        self.use_regularization = use_regularization

    def fit(self, X, y):
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

                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_predicted = sigmoid(linear_model)

                dw = (1 / len(X_batch)) * np.dot(X_batch.T, (y_predicted - y_batch))
                db = (1 / len(X_batch)) * np.sum(y_predicted - y_batch)

                if self.use_regularization:
                    dw += (self.regularization_strength * self.weights)  # Corrected regularization term

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db  # Corrected bias update logic

            loss = self.compute_loss(X, y)
            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}')            
            
            if np.allclose(prev_weights, self.weights, rtol=1e-05):  # Corrected stopping condition
                break

            prev_weights = self.weights

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        y_class_pred = [1 if i > 0.5 else 0 for i in y_predicted]  # Corrected equality condition
        return np.array(y_class_pred)