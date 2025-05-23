# -*- coding: utf-8 -*-
"""final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Y2Ip1D3JneDnezi9r1x7aQ7bNJVQN625
"""

# !pip install wandb -qU

# Log in to your W&B account
import wandb
import random
import math

wandb.login()
# API = ed57ccb8a48835266e803f637f8b571506709c5d

import numpy as np
import wandb
from keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 🔹 Load Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0

# 🔹 One-Hot Encoding for labels
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

# 🔹 Log Sample Images from Question 1
def log_sample_images():
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    plt.figure(figsize=(10, 10))
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_train[idx].reshape(28, 28), cmap='gray')
        plt.title(class_names[i])
        plt.axis('off')
    plt.tight_layout()
    wandb.log({"Sample Images": wandb.Image(plt)})
    plt.close()

# 🔹 Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 1e-4, 'max': 1e-2},
        'batch_size': {'values': [16, 32, 64]},
        'optimizer': {'values': ['sgd', 'adam', 'rmsprop']},
        'num_layers': {'values': [1, 2, 3]},
        'hidden_size': {'values': [64, 128, 256]},
    }
}

# 🔹 Initialize Sweep
sweep_id = wandb.sweep(sweep_config, project="da6401-assignment1")


# =========================
# 🔥 Neural Network Class 🔥
# =========================
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', weight_init='xavier'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = []
        self.biases = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            if self.weight_init == 'xavier':
                limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
                self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
            else:
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def activation_function(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            return x

    def activation_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - x ** 2
        else:
            return np.ones_like(x)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.activation_function(z))

        return self.activations[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        error = self.activations[-1] - y
        d_weights[-1] = np.dot(self.activations[-2].T, error) / m
        d_biases[-1] = np.sum(error, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(error, self.weights[i + 1].T) * self.activation_derivative(self.activations[i + 1])
            d_weights[i] = np.dot(self.activations[i].T, error) / m
            d_biases[i] = np.sum(error, axis=0, keepdims=True) / m

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * d_weights[i]
            self.biases[i] -= learning_rate * d_biases[i]

    def train(self, X, y, X_val, y_val, epochs, batch_size, learning_rate, optimizer='sgd'):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            # Validate on validation set
            val_preds = self.forward(X_val)
            val_loss = self.cross_entropy_loss(val_preds, y_val)
            val_acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1))

            # Evaluate on training set
            train_preds = self.forward(X)
            train_loss = self.cross_entropy_loss(train_preds, y)
            train_acc = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y, axis=1))

            # 🔹 Log results on wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            })

    def cross_entropy_loss(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))


# =========================
# 🔥 Training Function 🔥
# =========================
def train_sweep():
    run = wandb.init()  # Automatically gets config from sweep
    config = run.config

    # Extract hyperparameters
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    optimizer = config.optimizer

    print(f"🔹 Running Experiment: hidden={hidden_size}, layers={num_layers}, batch={batch_size}, opt={optimizer}")

    # Log sample images
    log_sample_images()

    # Initialize Model
    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=[hidden_size] * num_layers,
        output_size=10,
        activation='relu',
        weight_init='xavier'
    )

    # 🔹 Train Model & Log Results
    model.train(
        X_train, y_train_onehot,
        X_test, y_test_onehot,  # Validation set
        epochs=10,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer
    )

    # 🔹 Compute and Log Confusion Matrix
    y_pred_test = model.forward(X_test)
    y_pred_labels = np.argmax(y_pred_test, axis=1)
    y_true_labels = np.argmax(y_test_onehot, axis=1)

    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    wandb.log({"Confusion Matrix": wandb.Image(plt)})
    plt.close()

    print("✔ Training complete for this configuration")
    run.finish()  # Close wandb run


# =========================
# 🔥 Run Hyperparameter Sweep 🔥
# =========================
wandb.agent(sweep_id, train_sweep, count=5)