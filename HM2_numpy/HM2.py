import numpy as np


def normal_distribution(x, mu, sigma):
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    probability_density = coefficient * np.exp(exponent)
    return probability_density


def sigmoid_function(x):
    sigmoid_value = 1 / (1 + np.exp(-x))
    return sigmoid_value


def update_weights(w, X, y, y_hat, learning_rate=0.0005):
    n = len(X)
    for j in range(len(w)):
        gradient = np.sum((y_hat - y) * X[:, j]) / n
        w[j] -= learning_rate * gradient
    return w


def MSE(y, y_hat):
    return np.mean(np.square(y - y_hat))


def Binary_Cross_Entropy(y, y_hat):
    return -np.mean(y*np.log(y_hat) + (1 - y)*np.log(1-y_hat))

# Example usage normal_distribution:
mean_value = 0
std_deviation = 1
x_value = 0.5
print(normal_distribution(x_value, mean_value, std_deviation))


# Example usage sigmoid_function:
x_value = 2.0
print(sigmoid_function(x_value))


# Example usage update_weights:
w_vector = np.array([0.1, 0.2, 0.3])
X_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_vector = np.array([0, 1, 0])
y_hat_vector = np.random.rand(3)

print(update_weights(w_vector, X_matrix, y_vector, y_hat_vector))


# Example usage MSE:
target = np.array([3, 4, 5, 6, 7])
predictions = np.array([2.5, 3.5, 4.5, 5.5, 6.5])
print("Mean Squared Error:", MSE(target, predictions))


# Example usage Binary_Cross_Entropy:
target = np.array([0, 1, 1, 0, 1])
predictions = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
print("Cross-entropy Loss:", Binary_Cross_Entropy(target, predictions))
