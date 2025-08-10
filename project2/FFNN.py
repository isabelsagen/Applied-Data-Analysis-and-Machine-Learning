import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
 
class NeuralNetwork:
    def __init__(self,
                 X_data, Y_data, 
                 n_hidden_neurons=[50], 
                 n_output_neurons=1,
                 epochs=10, 
                 batch_size=100, 
                 eta=0.1, 
                 lmbd=0.0, 
                 activation='sigmoid',
                 task='regression'):
        # Initialize training data and network configuration
        self.X_data_full = X_data
        self.Y_data_full = Y_data
        
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]

        #Flexible hidden layer
        if isinstance(n_hidden_neurons, int):
            n_hidden_neurons = [n_hidden_neurons]
        self.n_hidden_neurons = n_hidden_neurons

        #Setting data parameters
        self.n_output_neurons = n_output_neurons
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.activation = activation

        # Initialize arrays to track metrics
        self.MSE = np.zeros(self.epochs)
        self.R2 = np.zeros(self.epochs)
        self.create_biases_and_weights()

    def create_biases_and_weights(self):
     #Initialize hidden layers, weights and biases
        self.hidden_weights = []
        self.hidden_biases = []
        
        layer_sizes = [self.n_features] + self.n_hidden_neurons
        for i in range(len(self.n_hidden_neurons)):
            self.hidden_weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.hidden_biases.append(np.zeros(layer_sizes[i+1]) + 0.01)

        self.output_weights = np.random.randn(self.n_hidden_neurons[-1], self.n_output_neurons) * 0.01
        self.output_bias = np.zeros(self.n_output_neurons) + 0.01

    def feed_forward(self):
     # Forward propagation: compute activations through the network
        a = self.X_data
        self.a_h = []
        self.z_h = []

     #Setting the activation functions
        for i in range(len(self.hidden_weights)):
            z = np.matmul(a, self.hidden_weights[i]) + self.hidden_biases[i]
            self.z_h.append(z)
            if self.activation == 'sigmoid':
                a = self.sigmoid(z)
            elif self.activation == 'ReLU':
                a = self.ReLU(z)
            elif self.activation == 'leaky_ReLU':
                a = self.leaky_ReLU(z)
            self.a_h.append(a)
        
        self.z_o = np.matmul(a, self.output_weights) + self.output_bias

     #Setting output functions based on regression or classification
        if self.task == 'classification':
            if self.n_output_neurons > 1:
                self.a_o = self.softmax(self.z_o)
            else:
                self.a_o = self.sigmoid(self.z_o)
        elif self.task == 'regression':
            self.a_o = self.z_o
        
    def backprop(self):
    # Backpropagation: compute gradients and update weights
        error_output = (self.a_o - self.Y_data) / self.Y_data.shape[0]

     #Compute gradient of output layer
        self.output_weights_gradient = np.matmul(self.a_h[-1].T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

     # Clip gradients to avoid explosion
        np.clip(self.output_weights_gradient, -1, 1, out=self.output_weights_gradient)
        np.clip(self.output_bias_gradient, -1, 1, out=self.output_bias_gradient)

     # Backpropagate through hidden layers
        error_hidden = error_output
        for i in reversed(range(len(self.hidden_weights))):
            activation_derivative = self.a_h[i] * (1 - self.a_h[i]) if self.activation == 'sigmoid' else (self.a_h[i] > 0).astype(float)
            
            error_hidden = error_hidden @ self.output_weights.T * activation_derivative if i == len(self.hidden_weights) - 1 else \
                           error_hidden @ self.hidden_weights[i+1].T * activation_derivative

            self.hidden_weights_gradient = np.matmul(self.a_h[i-1].T, error_hidden) if i > 0 else np.matmul(self.X_data.T, error_hidden)
            self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

            np.clip(self.hidden_weights_gradient, -1, 1, out=self.hidden_weights_gradient)
            np.clip(self.hidden_bias_gradient, -1, 1, out=self.hidden_bias_gradient)

            if self.lmbd > 0.0:
                self.hidden_weights_gradient += self.lmbd * self.hidden_weights[i]

            self.hidden_weights[i] -= self.eta * self.hidden_weights_gradient
            self.hidden_biases[i] -= self.eta * self.hidden_bias_gradient

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
    
    def train(self):
        data_indices = np.arange(self.n_inputs)
    
        for i in range(self.epochs):
         # Perform mini-batch gradient descent
            for _ in range(self.n_inputs // self.batch_size):
                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]
            
                self.feed_forward()
                self.backprop()
             
                # Compute metrics based on task
                if self.task == 'regression':
                    self.MSE[i] = self.mean_squared_error(self.Y_data, self.a_o)
                    self.R2[i] = self.r2_score(self.Y_data, self.a_o)
                elif self.task == 'classification':
                    self.MSE[i] = self.cross_entropy_loss(self.Y_data, self.a_o)
            
        return np.linspace(0, self.epochs, self.epochs), self.MSE, self.R2
    
    def predict(self, X):
     # Predict outputs for new data
        self.X_data = X
        self.feed_forward()
        if self.task == 'regression':
            return self.a_o
        elif self.task == 'classification':
            return (self.a_o > 0.5).astype(int) if self.n_output_neurons == 1 else np.argmax(self.a_o, axis=1)
    
    @staticmethod
    def sigmoid(x):
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_ReLU(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    @staticmethod
    def r2_score(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

