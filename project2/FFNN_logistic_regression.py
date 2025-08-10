import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from FFNN import NeuralNetwork  
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from itertools import product
from matplotlib import colors
import matplotlib as mpl
import pandas as pd

mpl.rcParams['font.size'] = 14

# Set a random seed for reproducibility
np.random.seed(123)

#-------------------------------------------------------------------------
#                     Breast Cancer Dataset
#-------------------------------------------------------------------------
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

nn = NeuralNetwork(X_data=X_train, Y_data=y_train, n_hidden_neurons=50, n_output_neurons=1, epochs=1000, 
                   batch_size=32, eta=0.01, lmbd=0.001, activation='sigmoid', task='classification')

# Train the neural network model and collect training history
epochs_array, mse, _ = nn.train()

# Predict on the test set and convert predictions to binary labels
y_pred = nn.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_labels)
print(f'Accuracy on test set: {accuracy}')

activations = ['sigmoid', 'ReLU', 'leaky_ReLU']
etas = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
lmbds = [0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
layer_counts = [1, 2, 3, 4, 5]
neurons_per_layer = [10, 30, 50, 80, 100]
results = []

# Loop through different combinations of activation, learning rate, and regularization
for activation in activations:
    for eta in etas:
        for lmbd in lmbds:
            nn = NeuralNetwork(
                X_data=X_train,
                Y_data=y_train,
                n_hidden_neurons=50,
                n_output_neurons=1,
                epochs=100, 
                batch_size=32,
                eta=eta,
                lmbd=lmbd,
                activation=activation,
                task='classification'
            )
            nn.train()
            y_pred = nn.predict(X_test)
            y_pred_labels = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_labels)

            results.append({
                'activation': activation,
                'eta': eta,
                'lambda': lmbd,
                'accuracy': accuracy
            })
          
# Convert results to a DataFrame for easier analysis
results = pd.DataFrame(results)

optimal_params = results.loc[results['accuracy'].idxmax()]

# Identify and print the optimal hyperparameters based on highest accuracy
print("Optimal Hyperparameters:")
print(f"Activation: {optimal_params['activation']}")
print(f"Learning Rate: {optimal_params['eta']}")
print(f"Regularization: {optimal_params['lambda']}")
print(f"Accuracy: {optimal_params['accuracy']}")

#----------------------------------------------------------------
#          Finding optimal layers and neurons per layer
#----------------------------------------------------------------

# Define range of hidden layers and neurons per layer
layer_counts = [1, 2, 3, 4, 5]
neurons_per_layer = [10, 30, 50, 80, 100]

accuracy_results = np.zeros((len(layer_counts), len(neurons_per_layer)))

# Loop through each combination of layer count and neurons per layer
for i, layers in enumerate(layer_counts):
    for j, neurons in enumerate(neurons_per_layer):
        nn = NeuralNetwork(X_train, y_train,
                           n_hidden_neurons=[neurons] * layers,
                           epochs=100,
                           batch_size=32,
                           eta=0.01,
                           lmbd=0.01,
                           task='classification')

        nn.train()
        y_pred = nn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[i, j] = accuracy

max_accuracy = np.max(accuracy_results)
max_accuracy_pos = np.unravel_index(np.argmax(accuracy_results), accuracy_results.shape)

# Plot heatmap to visualize the grid search results
plt.figure(figsize=(10, 10))
cax = plt.imshow(accuracy_results, cmap="rainbow", aspect='auto')
cbar = plt.colorbar(cax)
cbar.set_label("Accuracy")
plt.xticks(ticks=np.arange(len(neurons_per_layer)), labels=neurons_per_layer)
plt.yticks(ticks=np.arange(len(layer_counts)), labels=layer_counts)
plt.xlabel("Neurons per Layer")
plt.ylabel("Number of Layers")
#plt.title("Grid Search for Optimal Accuracy", fontsize=20)

plt.plot(max_accuracy_pos[1], max_accuracy_pos[0], 'w*', markersize=15, label=f'Max Accuracy = {max_accuracy:.4f}')
plt.legend()
plt.show()
plt.show()

#----------------------------------------------------------------
#              Adjusting hyperparameters
#----------------------------------------------------------------
# Initialize arrays to store train and test accuracy for each (eta, lambda) combination
train_accuracy = np.zeros((len(etas), len(lmbds)))
test_accuracy = np.zeros((len(etas), len(lmbds)))

for i, eta in enumerate(etas):
    for j, lmbd in enumerate(lmbds):
        nn = NeuralNetwork(X_train, y_train, eta=eta, lmbd=lmbd, epochs=1000, batch_size=32, 
                           n_hidden_neurons=50, n_output_neurons=1, task='classification')

        nn.train()
        train_pred = nn.predict(X_train)
        test_pred = nn.predict(X_test)

        train_accuracy[i][j] = accuracy_score(y_train, (train_pred > 0.5).astype(int))
        test_accuracy[i][j] = accuracy_score(y_test, (test_pred > 0.5).astype(int))

max_train_accuracy = np.max(train_accuracy)
max_train_pos = np.unravel_index(np.argmax(train_accuracy), train_accuracy.shape)

# Plot training accuracy heatmap
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(train_accuracy, cmap="rainbow", aspect='auto')
cbar = fig.colorbar(cax)
cbar.set_label("Accuracy")
#ax.set_title("Training Accuracy", fontsize=20)
ax.set_ylabel("Learning Rate")
ax.set_xlabel("Regularization")
ax.set_xticks(np.arange(len(lmbds)))
ax.set_yticks(np.arange(len(etas)))
ax.set_xticklabels(np.round(lmbds, 5))
ax.set_yticklabels(np.round(etas, 5))

ax.plot(max_train_pos[1], max_train_pos[0], 'w*', markersize=15, label=f'Max Accuracy = {max_train_accuracy:.4f}')
ax.legend()
plt.show()

max_test_accuracy = np.max(test_accuracy)
max_test_pos = np.unravel_index(np.argmax(test_accuracy), test_accuracy.shape)

# Plot test accuracy heatmap
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(test_accuracy, cmap="rainbow", aspect='auto')
cbar = fig.colorbar(cax)
cbar.set_label("Accuracy")
#ax.set_title("Test Accuracy", fontsize=20)
ax.set_ylabel("Learning Rate")
ax.set_xlabel("Regularization")
ax.set_xticks(np.arange(len(lmbds)))
ax.set_yticks(np.arange(len(etas)))
ax.set_xticklabels(np.round(lmbds, 5))
ax.set_yticklabels(np.round(etas, 5))


ax.plot(max_test_pos[1], max_test_pos[0], 'w*', markersize=15, label=f'Max Accuracy = {max_test_accuracy:.4f}')
ax.legend()
plt.show()

learning_rates = [0.01, 0.001, 0.0001]

accuracy_results = {}

for eta in learning_rates:
    nn = NeuralNetwork(X_train, y_train,
                       n_hidden_neurons=hidden_neurons,
                       n_output_neurons=1, 
                       epochs=400,
                       batch_size=32,
                       eta=eta,
                       lmbd=0.0,             
                       activation='sigmoid',
                       task='classification')

    epoch_accuracies = []

    for epoch in range(epochs):
        nn.train()
        train_pred = nn.predict(X_train)
        accuracy = accuracy_score(y_train, (train_pred > 0.5).astype(int))
        epoch_accuracies.append(accuracy)

    accuracy_results[eta] = epoch_accuracies

plt.figure(figsize=(10, 6))
for eta, acc_values in accuracy_results.items():
    plt.plot(range(epochs), acc_values, label=f'Learning Rate: {eta}')

plt.title("Training Accuracy vs. Epochs for Different Learning Rates")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
