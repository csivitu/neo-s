import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=50, batch_size=4, regularization_strength=0.01, use_regularization=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization_strength = regularization_strength
        self.use_regularization = use_regularization

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

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
                    dw += (self.regularization_strength / len(X_batch)) * self.weights

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

                if np.linalg.norm(dw) < 0.001:
                    break

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        y_class_pred = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_class_pred)

X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]])
y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1])

model = LogisticRegression(learning_rate=0.0001, epochs=5000, batch_size=2, regularization_strength=0.5)
model.fit(X_train, y_train)

predictions = model.predict(X_train)
print("Predicted classes:", predictions)

