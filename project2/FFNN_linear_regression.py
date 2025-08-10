import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from FFNN import NeuralNetwork  
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
from matplotlib import colors
import matplotlib as mpl
import pandas as pd

mpl.rcParams['font.size'] = 14


# Set random seed for reproducibility
np.random.seed(123)

np.random.seed(0) 
X = np.linspace(-2, 2, 100).reshape(-1, 1)  # Input data points
Y = 2 + 3 * X + 1.5 * X**2 #+ np.random.normal(0, 0.2, X.shape)  # Polynomial with noise

# Prepare for training with different learning rates
learning_rates = [0.001, 0.01, 0.1]  # Experiment with different learning rates
epochs = 1000  # Define the number of epochs
mse_results = {}

# Train the network for each learning rate and store MSE values
for eta in learning_rates:
    nn = NeuralNetwork(
        X_data=X,
        Y_data=Y,
        n_hidden_neurons=[10],  # 10 hidden neurons, you can adjust this
        n_output_neurons=1,
        epochs=epochs,
        batch_size=10,
        eta=eta,
        activation='sigmoid',
        task='regression'
    )
    
    epochs_range, mse, _ = nn.train()  # Train the model and collect MSE
    mse_results[eta] = mse

# Plotting MSE vs. epochs for each learning rate
plt.figure(figsize=(10, 6))
for eta, mse in mse_results.items():
    plt.plot(epochs_range, mse, label=f'Learning Rate: {eta}')

plt.xlabel("Epochs")
plt.ylabel("MSE")
#plt.title("MSE as a Function of Epochs for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()

#--------------------------------------------------------------------------
#   Test with simple second order polynomial and compre to Scikit-learn
#--------------------------------------------------------------------------
# Test with simple second order polynomial
n = 1000
X = np.random.randn(n, 1)
y = 2 + 3 * X + 1.5 * X**2  #+ np.random.randn(n, 1)
y = y.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train neural network on training data
model_nn = NeuralNetwork(X_train, y_train, task='regression', n_output_neurons=1, epochs=100, batch_size=32, eta=0.001, lmbd=0.01)
epochs, mse_nn, r2_nn = model_nn.train()

Y_pred_nn = model_nn.predict(X_test)
mse_test_nn = model_nn.mean_squared_error(y_test, Y_pred_nn)
r2_test_nn = model_nn.r2_score(y_test, Y_pred_nn)

epochs = 100
mlp_sklearn = MLPRegressor(hidden_layer_sizes=(50,), max_iter=epochs, alpha=0.01, learning_rate_init=0.01)
mlp_sklearn.fit(X_train, y_train.ravel())
Y_pred_sklearn = mlp_sklearn.predict(X_test)

mse_sklearn = mean_squared_error(y_test, Y_pred_sklearn)
r2_sklearn = r2_score(y_test, Y_pred_sklearn)

print("Neural Network MSE:", mse_test_nn)
print("Mean Squared Error with Scikit:", mse_sklearn)
print("Neural Network R2:", r2_test_nn)
print("R2 with Scikit:", r2_sklearn)
#-----------------------------------------------------------------------
#            Grid search for optimal number of layers and neurons
#-----------------------------------------------------------------------

layer_counts = [1, 2, 3, 4, 5]          
neurons_per_layer = [10, 30, 50, 80, 100]


mse_results = np.zeros((len(layer_counts), len(neurons_per_layer)))

for i, layers in enumerate(layer_counts):
    for j, neurons in enumerate(neurons_per_layer):
        nn = NeuralNetwork(X_train, y_train, 
                           n_hidden_neurons=[neurons] * layers, 
                           epochs=100, 
                           batch_size=32,
                           eta=0.01, 
                           lmbd=0.01, 
                           task='regression')

        nn.train()
        y_pred = nn.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_results[i, j] = mse

# Find and plot configuration with minimum MSE
min_mse = np.min(mse_results)
min_mse_pos = np.unravel_index(np.argmin(mse_results), mse_results.shape)

plt.figure(figsize=(8, 6))
plt.imshow(mse_results, cmap="rainbow", aspect='auto')
cbar = plt.colorbar()
cbar.set_label("MSE", rotation=270, labelpad=15)
plt.xticks(ticks=np.arange(len(neurons_per_layer)), labels=neurons_per_layer)
plt.yticks(ticks=np.arange(len(layer_counts)), labels=layer_counts)
#plt.title("MSE", fontsize=20)
plt.xlabel("Neurons per Layer")
plt.ylabel("Number of Layers")

plt.plot(min_mse_pos[1], min_mse_pos[0], 'w*', markersize=15, label=f'Min MSE = {min_mse:.4f}')
plt.legend()
plt.show()

#-----------------------------------------------------------------------
#            Grid search for optimal hyperparameters with Sigmoid
#-----------------------------------------------------------------------
eta_vals = np.logspace(-5, 0, 6) # Range of learning rates
lambda_vals = np.logspace(-5, 0, 6) # Range of regularization values

mse_results = np.zeros((len(lambda_vals), len(eta_vals)))

for i, lmbd in enumerate(lambda_vals):
    for j, eta in enumerate(eta_vals):
        nn = NeuralNetwork(X_train, y_train, 
                           n_hidden_neurons=50,
                           epochs=100, 
                           batch_size=32,
                           eta=eta, 
                           lmbd=lmbd, 
                           task='regression')
        
        nn.train()
        y_pred = nn.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_results[i, j] = mse

# Find and plot configuration with minimum MSE
min_mse = np.min(mse_results)
min_mse_pos = np.unravel_index(np.argmin(mse_results), mse_results.shape)

plt.figure(figsize=(10, 8))
plt.imshow(mse_results, cmap="rainbow", aspect='auto')
cbar = plt.colorbar()
cbar.set_label("MSE", rotation=270, labelpad=15)
plt.xticks(ticks=np.arange(len(eta_vals)), labels=np.round(eta_vals, 5))
plt.yticks(ticks=np.arange(len(lambda_vals)), labels=np.round(lambda_vals, 5))
#plt.title("MSE for Different Learning Rates and Regularization Parameters", fontsize=20)
plt.xlabel("Learning Rate")
plt.ylabel("Regularization")

plt.plot(min_mse_pos[1], min_mse_pos[0], 'w*', markersize=15, label=f'Min MSE = {min_mse:.4f}')
plt.legend()
plt.show()

#------------------------------------------------------------------------------------------
#     Grid search for optimal hyperparameters for different activation functions
#------------------------------------------------------------------------------------------

# Initialize arrays to store results for ReLU and Leaky ReLU
mse_results_relu = np.zeros((len(lambda_vals), len(eta_vals)))
mse_results_leaky_relu = np.zeros((len(lambda_vals), len(eta_vals)))

#ReLU
for i, lmbd in enumerate(lambda_vals):
    for j, eta in enumerate(eta_vals):
        nn = NeuralNetwork(X_train, y_train, 
                           n_hidden_neurons=50, 
                           epochs=100, 
                           batch_size=32,
                           eta=eta, 
                           lmbd=lmbd, 
                           activation='ReLU',
                           task='regression')
        
        nn.train()
        y_pred = nn.predict(X_test)
        if np.isnan(y_pred).any():
            mse_results_relu[i, j] = np.nan 
            print(f"NaN detected with eta={eta}, lambda={lmbd} for ReLU activation")
        else:
            mse = mean_squared_error(y_test, y_pred)
            mse_results_relu[i, j] = mse

#Leaky ReLU
for i, lmbd in enumerate(lambda_vals):
    for j, eta in enumerate(eta_vals):
        nn = NeuralNetwork(X_train, y_train, 
                           n_hidden_neurons=50, 
                           epochs=100, 
                           batch_size=32,
                           eta=eta, 
                           lmbd=lmbd, 
                           activation='leaky_ReLU',
                           task='regression')
        
        nn.train()
        y_pred = nn.predict(X_test)
        if np.isnan(y_pred).any():
            mse_results_leaky_relu[i, j] = np.nan
            print(f"NaN detected with eta={eta}, lambda={lmbd} for Leaky ReLU activation")
        else:
            mse = mean_squared_error(y_test, y_pred)
            mse_results_leaky_relu[i, j] = mse

min_mse_relu = np.nanmin(mse_results_relu)
min_mse_relu_pos = np.unravel_index(np.nanargmin(mse_results_relu), mse_results_relu.shape)

plt.figure(figsize=(10, 8))
plt.imshow(mse_results_relu, cmap="rainbow", aspect='auto')
cbar = plt.colorbar()
cbar.set_label("MSE", rotation=270, labelpad=15)
plt.xticks(ticks=np.arange(len(eta_vals)), labels=np.round(eta_vals, 5))
plt.yticks(ticks=np.arange(len(lambda_vals)), labels=np.round(lambda_vals, 5))
#plt.title("MSE for ReLU Activation")
plt.xlabel("Learning Rate")
plt.ylabel("Regularization")

plt.plot(min_mse_relu_pos[1], min_mse_relu_pos[0], 'w*', markersize=15, label=f'Min MSE = {min_mse_relu:.4f}')
plt.legend()
plt.show()

min_mse_leaky_relu = np.nanmin(mse_results_leaky_relu)
min_mse_leaky_relu_pos = np.unravel_index(np.nanargmin(mse_results_leaky_relu), mse_results_leaky_relu.shape)

plt.figure(figsize=(10, 8))
plt.imshow(mse_results_leaky_relu, cmap="rainbow", aspect='auto')
cbar = plt.colorbar()
cbar.set_label("MSE", rotation=270, labelpad=15)
plt.xticks(ticks=np.arange(len(eta_vals)), labels=np.round(eta_vals, 5))
plt.yticks(ticks=np.arange(len(lambda_vals)), labels=np.round(lambda_vals, 5))
#plt.title("MSE for Leaky ReLU Activation", fontsize=20)
plt.xlabel("Learning Rate", fontsize=20)
plt.ylabel("Regularization", fontsize=20)

# Add a star marker at the minimum MSE position
plt.plot(min_mse_leaky_relu_pos[1], min_mse_leaky_relu_pos[0], 'w*', markersize=15, label=f'Min MSE = {min_mse_leaky_relu:.4f}')
plt.legend()
plt.show()
