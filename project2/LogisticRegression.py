

import numpy as np
import sklearn.linear_model as skl_lin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
scaler = StandardScaler()
np.random.seed(1)


class Logistic_Regression:

    #initializing atributes
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_

    #extractiong the breast cancer data
    def generate_data(self): 
        x, y = load_breast_cancer(return_X_y=True)
        self.X_scaled = scaler.fit_transform(x)
        self.y = y.reshape(-1, 1)
        return self.X_scaled, self.y

    #sigmoid function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    #spliting the data in test and train sets
    def test_train_split(self, test_size_):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y, test_size = test_size_, random_state=123)
        

    # Gradient descent
    def gradient_descent(self, learning_rate, n_iterations, momentum, lambda_value, opt):
        """
        Gradient Descent with Momentum for logistic regression with both training and test accuracy tracking.
        If lambda=0, it performs logistic regression without regularization. If lambda>0, it includes L2 regularization.
        If momentum is 0, there is no momentum; otherwise, it uses the specified momentum factor.
        """

        # Add bias term and square term for quadratic regression
        X_b_train = np.c_[np.ones((self.X_train.shape[0], 1)), self.X_train, self.X_train**2]
        X_b_test = np.c_[np.ones((self.X_test.shape[0], 1)), self.X_test, self.X_test**2]
        
        beta = np.random.randn(X_b_train.shape[1], 1)  # Initialize weights
        delta_beta = np.zeros((X_b_train.shape[1], 1))  # Initialize momentum term
        eta = learning_rate
        accuracy_history_train = []
        accuracy_history_test = []


        for i in range(n_iterations):
            # Calculate gradient with regularization term
            gradient = -X_b_train.T @ (self.y_train - self.sigmoid(X_b_train @ beta)) + lambda_value * beta

            #if no optimisation method
            if opt == 0 or opt is None:
                delta_beta = -eta * gradient + momentum * delta_beta

            #if AdaGrad
            elif opt == "AdaGrad":
                new_eta = self.AdaGrad(1e-7, gradient, learning_rate)
                delta_beta = -new_eta * gradient + momentum * delta_beta
            
            #if RMSprop
            elif opt == "RMSprop":
                new_eta = self.RMSprop(1e-7, gradient, learning_rate, 0.9)
                delta_beta = -new_eta * gradient + momentum * delta_beta

            #if Adam
            elif opt == "Adam":
                new_eta = self.Adam(1e-8, gradient, learning_rate, i + 1, 0.9, 0.999)
                delta_beta = -new_eta

            # Update beta
            beta += delta_beta

            # Calculate predictions for training and test sets
            y_pred_proba_train = self.sigmoid(X_b_train @ beta)
            y_pred_train = (y_pred_proba_train >= 0.5).astype(int)

            y_pred_proba_test = self.sigmoid(X_b_test @ beta)
            y_pred_test = (y_pred_proba_test >= 0.5).astype(int)

            # Calculate accuracy for both training and test sets
            accuracy_history_train.append(accuracy_score(self.y_train, y_pred_train))
            accuracy_history_test.append(accuracy_score(self.y_test, y_pred_test))


        return accuracy_history_test




#----------------------------------------------------------------------------------------------------------

# SGD

    def stochastic_gradient_descent(self, learning_rate, momentum, lambda_value, n_epochs, batch_size, decay_rate, opt):
        """
        Stochastic Gradient Descent with mini-batches for logistic regression, tracking both training and test accuracy over epochs.
        """
        
        # Add bias term and square term for quadratic regression (for both train and test sets)
        X_b_train = np.c_[np.ones((self.X_train.shape[0], 1)), self.X_train, self.X_train**2]
        X_b_test = np.c_[np.ones((self.X_test.shape[0], 1)), self.X_test, self.X_test**2]


        beta = np.random.randn(X_b_train.shape[1], 1)  
        delta_beta = np.zeros((X_b_train.shape[1], 1))  

        # To store accuracy at each epoch
        accuracy_history_train = []  
        accuracy_history_test = []   
        eta = learning_rate

        for epoch in range(1, n_epochs + 1):
            # Optional: Decay the learning rate over epochs
            # eta = learning_rate / (1 + decay_rate * epoch)
            
            # Shuffle the training data at each epoch
            indices = np.random.permutation(len(self.y_train))
            X_b_train_shuffled = X_b_train[indices]
            y_train_shuffled = self.y_train[indices]

            for i in range(0, len(self.y_train), batch_size):
                # Select a mini-batch
                X_b_batch = X_b_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                # Calculate gradient with regularization term
                gradient = -X_b_batch.T @ (y_batch - self.sigmoid(X_b_batch @ beta)) + lambda_value * beta

                # Optimization method choice
                if opt == 0 or opt is None:
                    delta_beta = -eta * gradient + momentum * delta_beta

                #if AdaGrad
                elif opt == "AdaGrad":
                    new_eta = self.AdaGrad(1e-7, gradient, learning_rate)
                    delta_beta = -new_eta * gradient + momentum * delta_beta

                #if RMSprop
                elif opt == "RMSprop":
                    new_eta = self.RMSprop(1e-7, gradient, learning_rate, 0.9)
                    delta_beta = -new_eta * gradient + momentum * delta_beta

                #if Adam
                elif opt == "Adam":
                    new_eta = self.Adam(1e-8, gradient, learning_rate, epoch, 0.9, 0.999)
                    delta_beta = -new_eta

                # Update weights
                beta += delta_beta

            # Calculate predictions and accuracy for the training set
            y_pred_proba_train = self.sigmoid(X_b_train @ beta)
            y_pred_train = (y_pred_proba_train >= 0.5).astype(int)
            accuracy_history_train.append(accuracy_score(self.y_train, y_pred_train))

            # Calculate predictions and accuracy for the test set
            y_pred_proba_test = self.sigmoid(X_b_test @ beta)
            y_pred_test = (y_pred_proba_test >= 0.5).astype(int)
            accuracy_history_test.append(accuracy_score(self.y_test, y_pred_test))

        return accuracy_history_test 




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
    

