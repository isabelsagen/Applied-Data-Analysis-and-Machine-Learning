
from autograd import numpy as anp
import numpy as np
from sklearn import linear_model
import sklearn.linear_model as skl_lin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from autograd import grad
scaler = StandardScaler()
np.random.seed(1)


class Linear_Regression:

    #initializing atributes
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_

    #Mean square error
    def MSE(self, z_data, z_model):
        n = np.size(z_model)        
        return np.sum((z_data - z_model)**2)/n
    
    # loss function for autograd    
    def loss_function(self, beta, X, y):
        y_pred = anp.dot(X, beta)
        return anp.mean((y - y_pred) ** 2)

    #generating data with noise
    def generate_data(self, n, noise):
        np.random.seed(42)
        self.X = np.random.rand(n, 1)
        self.y = 2 + 3 * self.X + 1.5 * self.X**2 + noise * np.random.randn(n, 1)

        self.X_scaled = scaler.fit_transform(self.X)  # Scale the features

    #spliting dataset into test and training
    def test_train_split(self, test_size_):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y, test_size = test_size_, random_state=123)
         
    #plain gradient descent
    def gradient_descent(self, learning_rate, n_iterations, momentum, lambda_value, opt):
        # Add bias term and square term for quadratic regression
        X_b_train = np.c_[np.ones((self.X_train.shape[0], 1)), self.X_train, self.X_train**2]
        X_b_test = np.c_[np.ones((self.X_test.shape[0], 1)), self.X_test, self.X_test**2]
        
        # Initialize beta and delta_beta for gradient descent
        beta = np.random.randn(X_b_train.shape[1], 1)
        delta_beta = np.zeros((X_b_train.shape[1], 1))
        eta = learning_rate
        m = len(self.y_train)
        mse_history_train = []
        mse_history_test = []
        
        #when using autograd, remove comments
        #beta = anp.array(np.random.randn(X_b_train.shape[1], 1))
        #loss_grad = grad(self.loss_function)


        # Update the gradient calculation
        t = 0
        for i in range(n_iterations):
            t += 1

            #when using Autograd
            #gradient = loss_grad(beta, X_b_train, self.y_train) + 2 * lambda_value * beta

            #when not using autograd
            gradient = (2/m) * X_b_train.T @ (X_b_train @ beta - self.y_train) + 2 * lambda_value * beta

            #if using no optimalisation
            if (opt == 0 or None):
                delta_beta = -eta * gradient + momentum * delta_beta

            #if using AdaGrad
            elif opt == "AdaGrad":
                new_eta = self.AdaGrad(1e-7, gradient, learning_rate)
                delta_beta = -new_eta * gradient + momentum * delta_beta

            #if using RMSprop
            elif opt == "RMSprop":
                new_eta = self.RMSprop(1e-7, gradient, learning_rate, 0.9)
                delta_beta = -new_eta * gradient + momentum * delta_beta

            #If using Adam
            elif opt == "Adam":
                new_eta = self.Adam(1e-8, gradient, learning_rate, t, 0.9, 0.999)
                delta_beta = -new_eta

            #uppdating the beta values
            beta += delta_beta
            y_pred_train = X_b_train @ beta
            y_pred_test = X_b_test @ beta

            # Record MSE for both training and test sets
            mse_history_train.append(self.MSE(self.y_train, y_pred_train))
            mse_history_test.append(self.MSE(self.y_test, y_pred_test))

        return mse_history_test
    


    # Stochastic gradient descent
    def stochastic_gradient_descent(self, learning_rate, momentum, lambda_value, n_epochs, batch_size, decay_rate, opt):
        # Add bias term and square term for quadratic regression
        X_b_train = np.c_[np.ones((self.X_train.shape[0], 1)), self.X_train, self.X_train**2]
        X_b_test = np.c_[np.ones((self.X_test.shape[0], 1)), self.X_test, self.X_test**2]
        
        # Initialize beta and delta_beta 
        beta = np.random.randn(X_b_train.shape[1], 1)  
        delta_beta = np.zeros((X_b_train.shape[1], 1))

        # To store test MSE
        N = len(self.y_train)  
        mse_history_train = [] 
        mse_history_test = []  

        #when using autograd, remove comments
        #beta = anp.array(np.random.randn(X_b_train.shape[1], 1))
        #loss_grad = grad(self.loss_function)  

        #running over epochs
        for epoch in range(n_epochs):
            for i in range(0, N, batch_size):

                # Randomly select a batch of indices
                indices = np.random.choice(N, size=min(batch_size, N - i), replace=False)
                X_b_batch = X_b_train[indices]
                y_batch = self.y_train[indices]


                #when not using autograd
                gradient = (2 / len(indices)) * X_b_batch.T @ (X_b_batch @ beta - y_batch) + 2 * lambda_value * beta

                #when using Autograd
                #gradient = loss_grad(beta, X_b_train, self.y_train) + 2 * lambda_value * beta

                
                # Update delta_beta based on the selected optimization method

                #if using no optimalisation
                if opt == 0 or opt is None:
                    delta_beta = -learning_rate * gradient + momentum * delta_beta

                #if using AdaGrad
                elif opt == "AdaGrad":
                    new_eta = self.AdaGrad(1e-7, gradient, learning_rate)
                    delta_beta = -new_eta * gradient + momentum * delta_beta

                #if using RMSprop
                elif opt == "RMSprop":
                    new_eta = self.RMSprop(1e-7, gradient, learning_rate, 0.9)
                    delta_beta = -new_eta * gradient + momentum * delta_beta

                #if using Adam
                elif opt == "Adam":
                    new_eta = self.Adam(1e-8, gradient, learning_rate, epoch + 1, 0.9, 0.999)
                    delta_beta = -new_eta

                # Update beta
                beta += delta_beta

            # Predict on training and test sets after each epoch
            y_pred_train = X_b_train @ beta
            y_pred_test = X_b_test @ beta

            # Record MSE for both training and test sets
            mse_history_train.append(self.MSE(self.y_train, y_pred_train))
            mse_history_test.append(self.MSE(self.y_test, y_pred_test))

        return mse_history_test  # Return both histories
 
    

    #AdaGrad method
    def AdaGrad(self, delta, gradient, global_lr):
        if not hasattr(self, 'r'):
            self.r = np.zeros_like(gradient)

        #updating change in learning rate
        self.r += gradient ** 2
        new_lr = (global_lr/(delta + np.sqrt(self.r)))
        return new_lr

    #RMSprop method
    def RMSprop(self, delta, gradient, global_lr, rho):
        if not hasattr(self, 'r'):
            self.r = np.zeros_like(gradient)

        #updating change in learning rate
        self.r = (rho*self.r) + (1-rho)*(gradient ** 2)
        new_lr = (global_lr/(delta + np.sqrt(self.r))) 
        return new_lr

    #Adam method
    def Adam(self, delta, gradient, global_lr, t, rho_1, rho_2):
        if not hasattr(self, 'r'):
            self.r = np.zeros_like(gradient)
        if not hasattr(self, 's'):
            self.s = np.zeros_like(gradient)
        
        epsilon = 0.001 #suggested value by deep learning book

        self.s = rho_1 * self.s + (1 - rho_1) * gradient
        self.r = rho_2 * self.r + (1 - rho_2) * (gradient ** 2)
    
        s_hat = self.s / (1 - rho_1 ** t)
        r_hat = self.r / (1 - rho_2 ** t)

        #updating change in learning rate
        new_lr = (epsilon * s_hat)/(delta + np.sqrt(r_hat))
        return new_lr
    

