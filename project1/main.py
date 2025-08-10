
import myutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

mpl.rcParams['font.size'] = 13


#defining parameters
n = 50 #number of datapoints, feel free to change
m = 15 #max poly degree,  feel free to change
degrees = np.arange(1, m + 1)
noise_ = 1 #set 1 to include noise and something else to not
#z_type = 'Franke'
z_type = 'Terrain'
boot_n = 1000#  feel free to change
k_folds = 5 # feel free to change
split = 0.2 # feel free to change
lambdas = np.logspace(-4, 6, 31) # feel free to change
metode = 'OLS' # feel free to change


scaler = StandardScaler()
np.random.seed(666)

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)


x_scaled = scaler.fit_transform(x.reshape(-1, 1)).flatten() 
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten() 

LR = myutils.Linear_Regression(x_scaled, y_scaled)

#---------------------------------------------------------------------------
#                          OLS, MSE and R2 vs complexity
#---------------------------------------------------------------------------

''' 

m=15
#z_type = 'Terrain'
LR.make_z(noise_, z_type)

R2_test_OLS = np.zeros(m)
R2_train_OLS = np.zeros(m)
MSE_test_OLS = np.zeros(m)
MSE_train_OLS = np.zeros(m)

for i in range(1, m+1):
    LR.make_design(i)
    LR.test_train_split(split)
    LR.OLS()
    R2_test_OLS[i-1] = (LR.R2(LR.z_test, LR.z_test_pred))
    R2_train_OLS[i-1] = (LR.R2(LR.z_train, LR.z_train_pred))
    MSE_test_OLS[i-1] = (LR.MSE(LR.z_test, LR.z_test_pred))
    MSE_train_OLS[i-1] = (LR.MSE(LR.z_train, LR.z_train_pred))


plt.subplot(2, 1, 1)
plt.plot(degrees, MSE_test_OLS, label='MSE test', marker='o')
plt.plot(degrees, MSE_train_OLS, label='MSE train', marker='.')
plt.legend()
plt.grid()
plt.ylabel('MSE')

plt.subplot(2, 1, 2)
plt.plot(degrees, R2_test_OLS, label='$R^2$ test', marker='o')
plt.plot(degrees, R2_train_OLS, label='$R^2$ train', marker='.')
plt.xlabel('Polynomial Degree')
plt.ylabel('R²')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("terrain_OLS_n=50_MSE_R2.pdf", format="pdf")
plt.show()
'''

#---------------------------------------------------------------------------
#                 Bootstrap, test and train MSE vs complexity
#---------------------------------------------------------------------------

'''
metode = 'OLS'
m=15
degrees = np.arange(1, m + 1)
#z_type = 'Franke'
z_type = 'Terrain'
LR.make_z(noise_, z_type)
boot_n = 1000

MSE_test_boot = np.zeros(m)
MSE_train_boot = np.zeros(m)
bias_boot = np.zeros(m)
var_boot = np.zeros(m)

for i in range(1, m+1):
    LR.make_design(i)
    LR.Bootstrap(metode, boot_n, 0)
    MSE_test_boot[i-1] = LR.MSE_boot_test
    MSE_train_boot[i-1] = LR.MSE_boot_train

    
plt.plot(degrees, MSE_test_boot, label='MSE test', marker='o')
plt.plot(degrees, MSE_train_boot, label='MSE train', marker='.')
plt.grid()
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.tight_layout()
plt.legend()
plt.savefig("terrain_OLS_boot=1000_MSE_n=50_m=13.pdf", format="pdf")
plt.show()

'''

#---------------------------------------------------------------------------
#                     Bootstrap, Bias-variance vs complexity
#---------------------------------------------------------------------------

''' 
metode = 'OLS'
m=15
degrees = np.arange(1, m + 1)
z_type = 'Franke'
#z_type = 'Terrain'


LR.make_z(noise_, z_type)
boot_n = 1000

boot_OLS_error = np.zeros(m)
boot_OLS_bias = np.zeros(m)
boot_OLS_variance = np.zeros(m)

for i in range(1, m+1):
    LR.make_design(i)
    LR.Bootstrap(metode, boot_n, 0)
    boot_OLS_error[i-1] = (LR.MSE(LR.z_test, LR.z_test_pred))
    boot_OLS_bias[i-1] = np.mean((LR.z_test - np.mean(LR.z_test_pred, axis=0))**2)
    boot_OLS_variance[i-1] = np.mean(np.var(LR.z_test_pred, axis=0)) 
    
    
plt.plot(degrees, boot_OLS_bias, label='Bias', marker='o')
plt.plot(degrees, boot_OLS_error, label='Error', marker='o')
plt.plot(degrees, boot_OLS_variance, label='Variance', marker='.')

plt.grid()
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.tight_layout()
plt.legend()
plt.savefig("OLS_boot_bias_var.pdf", format="pdf")
plt.show()
'''

#---------------------------------------------------------------------------
#                            MSE, Bootstrap vs k-folds
#---------------------------------------------------------------------------

''' 
metode = 'OLS'
m=12
degrees = np.arange(1, m + 1)
#z_type = 'Franke'
z_type = 'Terrain'

LR.make_z(noise_, z_type)
boot_n = 1000

MSE_boot = np.zeros(m)
MSE_k_5 = np.zeros(m)
MSE_k_7 = np.zeros(m)
MSE_k_10 = np.zeros(m)

for i in range(1, m+1):
    LR.make_design(i)
    LR.Bootstrap(metode, boot_n, 0.0001)
    MSE_boot[i-1] = (LR.MSE(LR.z_test, LR.z_test_pred))
    LR.k_fold(metode, 5, 0.0001)
    MSE_k_5[i-1] = (LR.MSE(LR.z_test, LR.z_test_pred))

    LR.k_fold(metode, 10, 0.0001)
    MSE_k_10[i-1] = (LR.MSE(LR.z_test, LR.z_test_pred))


plt.plot(degrees, MSE_boot, label='Bootstrap', marker='o')
plt.plot(degrees, MSE_k_5, label='k = 5', marker='o')
plt.plot(degrees, MSE_k_10, label='k = 10', marker='o')

plt.grid()
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.tight_layout()
plt.legend()
plt.savefig("terrain_OLS_MSE_boot_k_n=50.pdf", format="pdf")
plt.show()
'''

#---------------------------------------------------------------------------
#                        Ridge/LASSO, MSE vs complexity vs lambda 
#---------------------------------------------------------------------------

'''

#z_type = 'Franke'
z_type = 'Terrain'
LR.make_z(noise_, z_type)
lambdas = np.logspace(-9, 1, 31)
m=15
degrees = np.arange(1, m + 1)
metode = 'Ridge'


#MSE_ridge = np.zeros((len(lambdas), m))
MSE_ridge = np.zeros(( m, len(lambdas)))


for i in range(1, m+1):
    for j in range(len(lambdas)):
        LR.make_design(i)
        LR.test_train_split(split)
        LR.k_fold(metode, 10, lambda[j])
        MSE_ridge[i-1, j] = (LR.MSE(LR.z_test, LR.z_test_pred))


plt.figure(figsize=(6, 4))
plt.imshow(MSE_ridge, aspect='auto', cmap='hot', interpolation='nearest')
plt.colorbar(label='MSE')
plt.ylabel('Polynomial Degree')
plt.xlabel(r'log10($\lambda$)')
plt.xticks(ticks=np.linspace(0, len(lambdas)-1, 11).astype(int), labels=[f'{np.log10(lambdas[i]):.1f}' for i in np.linspace(0, len(lambdas)-1, 11).astype(int)],rotation=45)
plt.yticks(ticks=np.linspace(0, m-1, 8).astype(int), labels=np.linspace(1, m, 8).astype(int))
min_mse = np.min(MSE_ridge)
min_index = np.unravel_index(np.argmin(MSE_ridge), MSE_ridge.shape)
min_x = min_index[1] 
min_y = min_index[0]  
print(min_mse)
# Mark the smallest MSE with a red star
plt.plot(min_x, min_y, 'r*', markersize=15, label=f'Smallest MSE = {min_mse:.4}')
plt.legend()  # Show the legend

plt.tight_layout()
plt.savefig("terrain_ridge.pdf", format="pdf")
plt.show()

#plt.imshow(MSE_ridge, cmap='hot', interpolation='nearest')
#plt.colorbar()
#plt.legend()
#plt.grid()
#plt.xlabel('Polynomial Degree')
#plt.tight_layout()
#plt.xlabel(r'log10($\lambda$)')
#plt.savefig("Franke_ridge.pdf", format="pdf")
#plt.show()
'''


