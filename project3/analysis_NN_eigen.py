import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
from NN_eigen import *

mpl.rcParams['font.size'] = 14

if __name__ == "__main__":
    #Set random seeds for reproducibility
    np.random.seed(123)
    tf.random.set_seed(123)

    # Define the symmetric matrix A from random matrix Q
    n = 6
    Q = np.random.rand(n, n)
    A = 1/2 * (Q + Q.T)

    # Define the initial x and t values
    t_max = 1e3
    x = np.random.normal(0, 1, n)
    t = np.linspace(0, t_max, 100)

    # Define the neural network architecture
    n_hidden = 1000
    layers = [{"nodes": n_hidden, "activation": "relu"}]

    # Train the neural network to find the max eigenvalues
    NN = NN_eigen(layers, A, t)
    epochs = 5000
    epochs_array = np.arange(epochs)
    lr = 0.001  # Learning rate
    eigvals, eigvecs, final_mse = NN.predict(epochs, x, t, lr)

    # Compute the eigenvalues and eigenvectors using numpy
    numpy_eigvals, numpy_eigvecs = np.linalg.eig(A)
    max_idx = np.argmax(numpy_eigvals)
    max_eigval = numpy_eigvals[max_idx]
    max_eigvec = numpy_eigvecs.T[max_idx]

    # Print the results
    print(f"Max numpy eigenvalue: {max_eigval}")
    print(f"Max neural network eigenvalue: {eigvals[-1]}")
    aligned_nn_eigvec = align_eigenvector_direction(max_eigvec, eigvecs[-1])
    print(f"Max numpy eigenvector: {max_eigvec}")
    print(f"Max neural network eigenvector: {aligned_nn_eigvec}")
    print(f"Final MSE for max eigenvalue: {final_mse}")

    # Plot the eigenvalues
    fig = plt.figure(figsize=(13, 10))
    plt.plot(epochs_array, eigvals, "r", label='Neural network')
    plt.hlines(numpy_eigvals, 0, epochs, colors='b', linestyles='dashed', label='Numerical diagonalization')
    plt.yticks(numpy_eigvals)
    plt.xlabel("epochs")
    plt.ylabel(f"eigenvalues")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the eigenvector components
    fig = plt.figure(figsize=(13, 10))
    plt.hlines(max_eigvec, 0, epochs, colors='b', linestyles='dashed', label="Numerical diagonalization")
    for i in range(n):
        plt.plot(epochs_array, eigvecs[:, i], label=f'i={i}')
    
    plt.ylabel(r'v$_{i}$')
    plt.xlabel("epochs")
    plt.ylim(min(max_eigvec) - 0.05, max(max_eigvec) + 0.05)
    plt.yticks(max_eigvec)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Test for min eigenvalue
    NN = NN_eigen(layers, A, t, max=False)
    eigvals, eigvecs, final_mse = NN.predict(epochs, x, t, lr)

    # Compute the eigenvalues and eigenvectors using numpy
    min_idx = np.argmin(numpy_eigvals)
    min_eigval = numpy_eigvals[min_idx]
    min_eigvec = numpy_eigvecs.T[min_idx]

    # Print the results
    print(f"Min numpy eigenvalue: {min_eigval}")
    print(f"Min neural network eigenvalue: {eigvals[-1]}")
    aligned_nn_eigvec = align_eigenvector_direction(min_eigvec, eigvecs[-1])
    print(f"Min numpy eigenvector: {min_eigvec}")
    print(f"Min neural network eigenvector: {aligned_nn_eigvec}")
    print(f"Final MSE for min eigenvalue: {final_mse}")

    # Plot the eigenvalues
    fig = plt.figure(figsize=(13, 10))
    plt.plot(epochs_array, eigvals, "r", label='Neural network')
    plt.hlines(numpy_eigvals, 0, epochs, colors='b', linestyles='dashed', label='Numerical diagonalization')
    plt.xlabel("epochs")
    plt.ylabel(f"eigenvalues")
    plt.yticks(numpy_eigvals)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the eigenvector components
    fig = plt.figure(figsize=(13, 10))
    plt.hlines(min_eigvec, 0, epochs, colors='b', linestyles='dashed', label="Numerical diagonalization")
    for i in range(n):
        plt.plot(epochs_array, eigvecs[:, i], label=f'i={i}')
    
    plt.ylabel(r'v$_{i}$')
    plt.xlabel("epochs")
    plt.ylim(min(min_eigvec) - 0.05, max(min_eigvec) + 0.05)
    plt.yticks(min_eigvec)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Test with set x(0)
    # Start from the smallest eigenvalue eigenvector
    x0 = min_eigvec
    NN = NN_eigen(layers, A, t, max=False)
    eigvals, eigvecs, final_mse = NN.predict(epochs, x0, t, lr)

    fig = plt.figure(figsize=(13, 10))
    plt.plot(epochs_array, eigvals, "r", label='Neural network')
    plt.hlines(numpy_eigvals, 0, epochs, colors='b', linestyles='dashed', label='Numerical diagonalization')
    plt.xlabel("epochs")
    plt.ylabel(f"eigenvalues")
    plt.yticks(numpy_eigvals)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Start from max eigenvalue eigenvector
    x0 = max_eigvec
    NN = NN_eigen(layers, A, t)
    eigvals, eigvecs, final_mse = NN.predict(epochs, x0, t, lr)

    fig = plt.figure(figsize=(13, 10))
    plt.plot(epochs_array, eigvals, "r", label='Neural network')
    plt.hlines(numpy_eigvals, 0, epochs, colors='b', linestyles='dashed', label='Numerical diagonalization')
    plt.xlabel("epochs")
    plt.ylabel(f"eigenvalues")
    plt.yticks(numpy_eigvals)
    plt.legend()
    plt.tight_layout()
    plt.show()

