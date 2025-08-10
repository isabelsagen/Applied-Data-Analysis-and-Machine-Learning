import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Set the default font size for plots
mpl.rcParams['font.size'] = 22

# Define a deep neural network to solve the diffusion equation
class DeepResNet(tf.keras.Model):
    def __init__(self, layers, input_dim):
        super(DeepResNet, self).__init__()
        self.layers_list = []
        for layer in layers:
            self.layers_list.append(tf.keras.layers.Dense(layer["nodes"], activation=layer["activation"], dtype=tf.float64))
        self.layers_list.append(tf.keras.layers.Dense(1, name="output", dtype=tf.float64))
        
    #Forward pass for the model
    def call(self, inputs):
        for layer in self.layers_list:
            inputs = layer(inputs)
        return inputs
        
#Define the inital consition function for the PDE
def initial_condition(x):
    return tf.sin(np.pi * x)

#Construct the trial solution based on the neural network's output
def g(model, x, t):
    input_tensor = tf.concat([x, t], axis=1)
    model_output = model(input_tensor, training=True)
    trial_solution = (1 - t) * initial_condition(x) + x * (1 - x) * t * model_output
    return trial_solution

#Compute the loss function(MSE) for the PDE
def loss(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        trial = g(model, x, t)
        d_g_dx = tape.gradient(trial, x)
        d_g_dt = tape.gradient(trial, t)
    d2_g_d2x = tape.gradient(d_g_dx, x)
    MSE = tf.reduce_mean(tf.square(d2_g_d2x - d_g_dt))
    return MSE

#Prepare input data by creating a grid of x and t values
def prepare_data(x, t):
    X, T = tf.meshgrid(tf.convert_to_tensor(x, dtype=tf.float64), tf.convert_to_tensor(t, dtype=tf.float64))
    x = tf.reshape(X, [-1, 1])
    t = tf.reshape(T, [-1, 1])
    return x, t

#Perform a single training step
def train_step(model, x, t, optimizer):
    with tf.GradientTape() as tape:
        cost_value = loss(model, x, t)
    grads = tape.gradient(cost_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return cost_value

#Train the neural network over multiple epochs
def train(model, x, t, epochs, optimizer):
    learning_list = []
    for epoch in range(epochs):
        cost_value = train_step(model, x, t, optimizer)
        learning_list.append(cost_value.numpy())
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {cost_value.numpy()}")
    return learning_list
    
#Plot the neural network's predictions and learning progress
def plot_nn(model, x, t, n_x, n_t, learning_list):
    X_test, T_test = np.meshgrid(np.linspace(0.0, 1.0, n_x), np.linspace(0.0, 1.0, n_t))
    X_flattened = X_test.flatten()
    T_flattened = T_test.flatten()
    input_tensor = tf.convert_to_tensor(np.column_stack((X_flattened, T_flattened)), dtype=tf.float64)
    predictions_flattened = g(model, X_flattened[:, None], T_flattened[:, None]).numpy()
    predictions_reshaped = predictions_flattened.reshape(X_test.shape)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 10))
    ind = [int(1 * (n_t / 50)), int(10 * (n_t / 50)), int(20 * (n_t / 50)), int(35 * (n_t / 50))]

    for i in range(4):
        col = i % 2
        row = i // 2
        hd = np.linspace(0, 1, 500)
        ax[row, col].plot(x, predictions_reshaped[ind[i], :], '-', lw=2, label="NN")
        ax[row, col].plot(hd, analytic(hd, t[ind[i]]), "--", lw=2, label="Analytic")
        ax[row, col].text(0.3, 0.5, "MSE = %.2g" % mse(predictions_reshaped[ind[i], :], analytic(x, t[ind[i]])))
        ax[row, col].set_ylim((-0.01, 1.05))
        ax[row, col].set_xlim((-0.01, 1.01))
        ax[row, col].set_title(f"t={t[ind[i]]:.3f}")

    ax[0, 1].legend()
    fig.supylabel("f(x,t)")
    fig.supxlabel("x")
    fig.tight_layout()
    plt.show()

    fig_mse, ax_mse = plt.subplots(figsize=(12, 10))
    plt.plot(range(len(learning_list)), learning_list, lw=2)
    plt.xlim((0, len(learning_list)))
    plt.ylim((1e-6, 10))
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.yscale('log')
    fig_mse.tight_layout()
    plt.show()

#Calculate the Mean Squared Error (MSE) at specific time points
def calculate_mse_at_time(model, x, t_values):
    mse_values = []
    for t in t_values:
        t_tensor = tf.constant(t, dtype=tf.float64, shape=(x.shape[0], 1))
        predictions = g(model, x, t_tensor)
        exact_solution = analytic(x.numpy(), t)
        mse_value = mse(predictions.numpy(), exact_solution)
        mse_values.append(mse_value)
    return mse_values

#Exact solution of the diffusion equation for comparison
def analytic(x, t):
    return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

#Compute MSE between two arrays
def mse(x, y):
    return np.mean((x - y)**2)
