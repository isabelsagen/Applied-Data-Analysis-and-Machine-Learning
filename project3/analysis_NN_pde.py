import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl 
from NN_pde import *

mpl.rcParams['font.size'] = 22

#Train the neural network to solve the diffusion equation and plot the results
def solve(n_epochs, learning_rate, delta_x, delta_t):
    n_hidden = 100 #define the numer of hidden nodes per layer
    
    #Define spatial domains
    x0 = 0
    x1 = 1
    n_x = int((x1 - x0) / delta_x) + 1
    n_t = int((1 - 0) / delta_t) + 1

    #Generate grid points for x and y
    x = np.linspace(x0, x1, n_x)
    t = np.linspace(x0, x1, n_t)

    #Define the architecture
    layers = [{"nodes": n_hidden, "activation": "tanh"},
              {"nodes": n_hidden, "activation": "sigmoid"}]
    model = DeepResNet(layers=layers, input_dim=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    x_tensor, t_tensor = prepare_data(x, t)
    learning_list = train(model, x_tensor, t_tensor, n_epochs, optimizer)
    plot_nn(model, x, t, n_x, n_t, learning_list)
    return model, x_tensor

#Solve the diffusion equation for different grid resolutions and plot MSE over time
def solve_and_plot_mse(n_epochs, learning_rate, delta_pairs):
    t_values = np.linspace(0, 1, 100)

    for delta_x, delta_t in delta_pairs:
        n_x = int(1 / delta_x) + 1
        n_t = int(1 / delta_t) + 1

        x = np.linspace(0, 1, n_x)
        t = np.linspace(0, 1, n_t)

        layers = [{"nodes": 100, "activation": "tanh"},
                  {"nodes": 100, "activation": "sigmoid"}]
        model = DeepResNet(layers=layers, input_dim=2)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        x_tensor, t_tensor = prepare_data(x, t)
        train(model, x_tensor, t_tensor, n_epochs, optimizer)
        mse_values = calculate_mse_at_time(model, x_tensor[:, 0:1], t_values)
        plt.plot(t_values, mse_values, label=f'Δx={delta_x}, Δt={delta_t}')

    plt.xlabel('Time (t)')
    plt.ylabel('MSE')
    plt.title('MSE vs Time for different Δx and Δt values')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    
    #Hyperparameters
    n_epochs = 5000
    learning_rate = 0.01

    # Specific values for solve function
    print("Training and plotting NN solutions with Δx=0.01 and Δt=0.005...")
    solve(n_epochs, learning_rate, delta_x=0.01, delta_t=0.005)

    print("Training and plotting NN solutions with Δx=Δt=0.02...")
    solve(n_epochs, learning_rate, delta_x=0.02, delta_t=0.02)

    # Values for solve_and_plot_mse function
    delta_pairs = [(0.01, 0.005), (0.02, 0.02), (0.03, 0.03), (0.04, 0.04)]
    print("Plotting MSE vs Time for specific Δx and Δt values...")
    solve_and_plot_mse(n_epochs, learning_rate, delta_pairs)

if __name__ == "__main__":
    main()
