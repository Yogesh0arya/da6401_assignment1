GitHub Repository:
https://github.com/Yogesh0arya/da6401_assignment1

Report link: 
[https://wandb.ai/yogesh084arya-indian-institute-of-technology-madras/DA6401-Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTcwOTgwNw/edit?draftId=VmlldzoxMTcwOTgwNw==](https://wandb.ai/yogesh084arya-indian-institute-of-technology-madras/DA6401-Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTcwOTgwNw)

# README for DA6401 Assignment 1
This repository contains the code and report for DA6401 Assignment 1, which involves implementing a feedforward neural network from scratch, training it on the Fashion-MNIST dataset, and tracking experiments using Weights & Biases (wandb). The goal of the assignment is to implement backpropagation, experiment with various optimization algorithms, and perform hyperparameter tuning to achieve the best validation accuracy.

# Table of Contents
1. Project Overview
2. Installation
3. Usage
4. Code Structure
5. Hyperparameter Tuning
6. Results

# Project Overview
The project involves the following tasks:

1. Dataset Visualization: Plotting one sample image from each class in the Fashion-MNIST dataset.

2. Feedforward Neural Network: Implementing a flexible feedforward neural network with support for multiple hidden layers and neurons.

3. Backpropagation: Implementing backpropagation with support for various optimization algorithms (SGD, Momentum, Nesterov, RMSProp, Adam, Nadam).

4. Hyperparameter Tuning: Using wandb's sweep functionality to find the best hyperparameters for the model.

5. Evaluation: Reporting the best validation and test accuracy, plotting the confusion matrix, and comparing cross-entropy loss with squared error loss.

6. Recommendations: Providing recommendations for hyperparameter configurations for the MNIST dataset based on learnings from Fashion-MNIST.

6. Installation
To run the code, you need to install the required dependencies. You can do this by running the following command:


# Installation
To run the code, you need to install the required dependencies. You can do this by running the following command:

pip install -r requirements.txt


# Dependencies
1. Python 3.x
2. NumPy
3. Pandas
4. Matplotlib
5. Seaborn
6. Keras (for dataset loading)
7. Weights & Biases (wandb)

# Code Structure
The repository is organized as follows:

da6401_assignment1/

├── main.py               # Main script to train the model

├── requirements.txt       # List of dependencies

├── README.md              # This file is Readme file

├── report.pdf             # This file is report

└── wandb/                 # Wandb logs and sweep configurations

├── .git                   # This file is of git

├── Problem statement.pdf  # This file is of problem statement



# Hyperparameter Tuning
Hyperparameter tuning was performed using wandb's sweep functionality. The following hyperparameters were tuned:
1. Number of epochs: 5, 10

2. Number of hidden layers: 3, 4, 5

3. Hidden layer size: 32, 64, 128

4. Learning rate: 1e-3, 1e-4

5. Optimizer: SGD, Momentum, Nesterov, RMSProp, Adam, Nadam

6. Batch size: 16, 32, 64

7. Weight initialization: Random, Xavier

8. Activation function: Sigmoid, Tanh, ReLU

9. The best configuration achieved a validation accuracy of 85.62%.

# Results
- Best Validation Accuracy: 85.62%

- Test Accuracy: 85.62%

- Confusion Matrix: Generated using the best model.

- Loss Comparison: Cross-entropy loss outperformed squared error loss in terms of convergence and accuracy.
