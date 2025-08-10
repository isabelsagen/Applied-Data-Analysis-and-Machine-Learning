

import numpy as np
from sklearn import linear_model
import sklearn.linear_model as skl_lin
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from imageio import imread
scaler = StandardScaler()



class Linear_Regression:

    #initializing atributes
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_

    #making the design matrix from the x and y vectors
    def make_design(self, m):
        num_points = len(self.x)
        design_mat = np.zeros((num_points, int((m+1)*(m+2)/2))) 

        #looping to make the polynomial terms and inserting numerical values
        for k in range(num_points):
            index = 0
            for i in range(m + 1):
                for j in range(m + 1):  
                    if (j + i > m):
                        break
                    design_mat[k, index] = (self.x[k] ** i) * (self.y[k] ** j)
                    index += 1
        self.design = design_mat
        return self.design
    
    #making the Franke or Terrain data
    def make_z(self, noise, type):

        #making Franke function
        if(type == 'Franke'):
            X, Y = np.meshgrid(self.x, self.y)

            #definition of Franke, borrowed from lecture notes
            term1 = 0.75*np.exp(-(0.25*(9*X-2)**2) - 0.25*((9*Y-2)**2))
            term2 = 0.75*np.exp(-((9*X+1)**2)/49.0 - 0.1*(9*Y+1))
            term3 = 0.5*np.exp(-(9*X-7)**2/4.0 - 0.25*((9*Y-3)**2))
            term4 = -0.2*np.exp(-(9*X-4)**2 - (9*Y-7)**2)
            self.z = (term1 + term2 + term3 + term4)

            #noise term with normal distribution
            noise_term = 0.1*np.random.normal(0, 1, size = self.z.shape) #2D

            #with noise
            if (noise == 1):
                self.z = self.z + noise_term
                return self.z
            #without noise
            else:
                return self.z
        
        #extracting terrain data and scaling
        if(type == 'Terrain'):
            terrain = imread('SRTM_data_Norway_1.tif')
            terrain = terrain[:len(self.x),:len(self.x)]
            unscaled = np.reshape(terrain, (len(terrain), len(terrain)))
            self.z = scaler.fit_transform(unscaled)
            return self.z
        
        else: 
            return 'insert Franke or Terrain'

    #spliting the data into test and train sets
    def test_train_split(self, test_size_):
        self.des_train, self.des_test, self.z_train, self.z_test = train_test_split(self.design, self.z, test_size = test_size_, random_state=123)
         
    #finding R^2
    def R2(self, z_data, z_model):
        return 1 - np.sum((z_data - z_model) ** 2) / np.sum((z_data - np.mean(z_data)) ** 2)

    #Mean square error
    def MSE(self, z_data, z_model):
        n = np.size(z_model)        
        return np.sum((z_data - z_model)**2)/n

    #ordinary least square regression
    def OLS(self):
        self.beta_OLS = np.linalg.pinv(self.des_train.T @ self.des_train) @ self.des_train.T @ self.z_train
        self.z_train_pred = self.des_train @ self.beta_OLS
        self.z_test_pred = self.des_test @ self.beta_OLS

    #Ridge regression
    def Ridge(self, lambda_value):
        self.beta_Ridge = np.linalg.pinv(self.des_train.T @ self.des_train + lambda_value * np.eye(self.des_train.shape[1])) @ self.des_train.T @ self.z_train
        self.z_train_pred = self.des_train @ self.beta_Ridge
        self.z_test_pred = self.des_test @ self.beta_Ridge

    #LASSO regression using sklearns linear_model.LASSO
    def LASSO(self, lambda_value):
        RegLasso = linear_model.Lasso(alpha = lambda_value, fit_intercept= False, max_iter = 1000, tol = 1e-3)  
        RegLasso.fit(self.des_train, self.z_train)  
        self.z_train_pred = RegLasso.predict(self.des_train) 
        self.z_test_pred = RegLasso.predict(self.des_test) 
        self.beta_LASSO = RegLasso.coef_

    #Bootstrap method for resampeling
    def Bootstrap(self, method, bootstraps, lambda_value):
        self.test_train_split(0.2)

        #storing data for resampeling
        design_train_boot = self.des_train
        z_train_boot = self.z_train
        z_test_boot = self.z_test

        #making arrays for storing data
        z_pred_train_bootstrap = np.zeros((bootstraps, len(self.z_train), len(self.z_train[0]))) 
        z_pred_test_bootstrap = np.zeros((bootstraps, len(self.z_test), len(self.z_test[0])))  
        MSE_test_list = np.zeros(bootstraps)

        #bootstrap for ordinary least squares
        if (method == "OLS"):
            for i in range(bootstraps):
                self.des_train, self.z_train = resample(design_train_boot, z_train_boot)
                self.OLS()
                z_pred_train_bootstrap[i, :, :] = self.z_train_pred
                z_pred_test_bootstrap[i, :, :] = self.z_test_pred
                MSE_test_list[i] = self.MSE(z_test_boot, z_pred_test_bootstrap[i, :])
            
            self.des_train = design_train_boot
            self.z_train = z_train_boot
            self.z_train_pred = np.mean(z_pred_train_bootstrap, axis=0)
            self.z_test_pred = np.mean(z_pred_test_bootstrap, axis=0)

            self.OLS()

        #bootstrap for ordinary ridge regression
        if (method == "Ridge"):
            for i in range(bootstraps):
                self.des_train, self.z_train = resample(design_train_boot, z_train_boot)
                self.Ridge(lambda_value)
                z_pred_train_bootstrap[i, :, :] = self.z_train_pred
                z_pred_test_bootstrap[i, :, :] = self.z_test_pred
                MSE_test_list[i] = self.MSE(z_test_boot, z_pred_test_bootstrap[i, :])
            
            self.des_train = design_train_boot
            self.z_train = z_train_boot
            self.z_train_pred = np.mean(z_pred_train_bootstrap, axis=0)
            self.z_test_pred = np.mean(z_pred_test_bootstrap, axis=0)

            self.Ridge(lambda_value)

        #bootstrap for ordinary LASSO regression
        if (method == "LASSO"):
            for i in range(bootstraps):
                self.des_train, self.z_train = resample(design_train_boot, z_train_boot)
                self.LASSO(lambda_value)
                z_pred_train_bootstrap[i, :, :] = self.z_train_pred
                z_pred_test_bootstrap[i, :, :] = self.z_test_pred
                MSE_test_list[i] = self.MSE(z_test_boot, z_pred_test_bootstrap[i, :])
            
            self.des_train = design_train_boot
            self.z_train = z_train_boot
            self.z_train_pred = np.mean(z_pred_train_bootstrap, axis=0)
            self.z_test_pred = np.mean(z_pred_test_bootstrap, axis=0)

            self.LASSO(lambda_value)

        #defining statistical values after resampeling and regression
        self.MSE_boot_train = self.MSE(z_train_boot, self.z_train_pred)
        self.MSE_boot_test = self.MSE(z_test_boot, self.z_test_pred)       
        self.bias_boot = np.mean((self.z_test-np.mean(self.z_test_pred,axis=0, keepdims=True))**2)
        self.var_boot = np.mean(np.var(self.z_test_pred,axis=0, keepdims=True))


    #resampeling using cross validation k-folds from sklearn.model_selection's KFold
    def k_fold(self, method, k, lambda_value):

        #defining the k folds
        kfold = KFold(n_splits = k, shuffle=True, random_state=123)

        #lists for storing MSE
        MSE_train_fold = []
        MSE_test_fold = []

        #doing the resampeling
        for train_index, test_index in kfold.split(self.design):
            des_train_fold, des_test_fold = self.design[train_index], self.design[test_index]
            z_train_fold, z_test_fold = self.z[train_index], self.z[test_index]
        
            self.des_train = des_train_fold
            self.z_train = z_train_fold
            self.des_test = des_test_fold
            self.z_test = z_test_fold

            #deciding the regression method
            if (method == "OLS"):
                self.OLS()

            if (method == "Ridge"):
                self.Ridge(lambda_value)

            if (method == "LASSO"):
                self.LASSO(lambda_value)

            #storing MSE values
            mse_train = self.MSE(self.z_train, self.z_train_pred)
            mse_test = self.MSE(self.z_test, self.z_test_pred)

            MSE_train_fold.append(mse_train)
            MSE_test_fold.append(mse_test)
        
        #final MSEs
        self.MSE_train_k_fold = np.mean(MSE_train_fold)
        self.MSE_test_k_fold = np.mean(MSE_test_fold)

            







         
            









