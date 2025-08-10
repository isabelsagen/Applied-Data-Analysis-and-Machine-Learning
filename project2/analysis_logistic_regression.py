
import LogisticRegression
from LogisticRegression import Logistic_Regression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

mpl.rcParams['font.size'] = 14

n = 1000 #constant throughout the runs
learning_rate = 0.01
n_iterations = 1000
momentum = 0.9 
lambda_value = 0.00
n_epochs = 1000
batch_size = 10
noise = 0.1
opt = 0


scaler = StandardScaler()
np.random.seed(666)

x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)


x_scaled = scaler.fit_transform(x.reshape(-1, 1)).flatten() 
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten() 

LR = LogisticRegression.Logistic_Regression(x_scaled, y_scaled)

model = Logistic_Regression(np.linspace(0, 1, n), np.linspace(0, 1, n))
model.generate_data()
model.test_train_split(0.3)  




#----------------------------------------------------------------------------------------------------------
#                        Plain GD, with and without momentum,  MSE vs learning rate
#----------------------------------------------------------------------------------------------------------

'''
learning_rate = np.logspace(-5, -1, 30)
MSE_wo_mom= np.zeros_like(learning_rate)
MSE_w_mom = np.zeros_like(learning_rate)
MSE_wo_mom_s = np.zeros_like(learning_rate)
MSE_w_mom_s = np.zeros_like(learning_rate)

for i in range(len(learning_rate)):
    MSE_wo_mom[i] = model.gradient_descent(learning_rate[i], n_iterations, 0, lambda_value, 0)[-1]
    MSE_w_mom[i] = model.gradient_descent(learning_rate[i], n_iterations, momentum, lambda_value, 0)[-1]
    MSE_wo_mom_s[i] = model.stochastic_gradient_descent(learning_rate[i], 0, 0, n_epochs, batch_size, 0, 0)[-1]
    MSE_w_mom_s[i] = model.stochastic_gradient_descent(learning_rate[i], 0.9, 0, n_epochs, batch_size, 0, 0)[-1]
    print(i)
    

plt.plot( learning_rate, MSE_wo_mom, label=r'GD, $\gamma  = 0$')
plt.plot(learning_rate, MSE_w_mom, label=r'GD, $\gamma  = 0.9$')
plt.plot( learning_rate, MSE_wo_mom_s, label=r'SGD, $\gamma  = 0$')
plt.plot(learning_rate, MSE_w_mom_s, label=r'SGD, $\gamma  = 0.9$')
plt.xlabel(r'$\eta$')
plt.ylabel('Accuracy')
plt.xscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("log_GD_SGD_MSE_lerningrate.pdf", format="pdf")
plt.show()
'''

#----------------------------------------------------------------------------------------------------------
#                         Plain GD, MSE vs iterations
#----------------------------------------------------------------------------------------------------------

'''   
mse_ols_wo_mom_001 = model.gradient_descent(0.01, n_iterations, 0, lambda_value, 0)
mse_ols_wo_mom_0001 = model.gradient_descent(0.001, n_iterations, 0, lambda_value, 0)
mse_ols_wo_mom_00001 = model.gradient_descent(0.0001, n_iterations, 0, lambda_value, 0)
mse_ols_wo_mom_00001_m = model.gradient_descent(0.0001, n_iterations, momentum, lambda_value, 0)


plt.plot(mse_ols_wo_mom_001,label=r'$\eta =0.01$')
plt.plot(mse_ols_wo_mom_0001,label=r'$\eta =0.001$')
plt.plot(mse_ols_wo_mom_00001,label=r'$\eta =0.0001$')
plt.plot(mse_ols_wo_mom_00001_m,label=r'$\eta =0.0001, \gamma = 0.9$')

plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.grid()
plt.xlim(0, 1000)
plt.legend()
plt.tight_layout()
plt.savefig("log_GD_MSE.pdf", format="pdf")
plt.show()
'''

#----------------------------------------------------------------------------------------------------------
#                  Stochastic GD, MSE vs epochs for different optimalisations
#----------------------------------------------------------------------------------------------------------

''' 
mse_history = model.stochastic_gradient_descent(learning_rate, 0, 0, n_epochs, batch_size, 0, 0)
mse_history_adagrad = model.stochastic_gradient_descent(learning_rate, 0, 0, n_epochs, batch_size, 0, 'AdaGrad')
mse_history_RMS = model.stochastic_gradient_descent(learning_rate, 0, 0, n_epochs, batch_size, 0, 'RMSprop')
mse_history_adam = model.stochastic_gradient_descent(learning_rate, 0, 0, n_epochs, batch_size, 0, 'Adam')

plt.plot(mse_history_adagrad,label='Adagrad')
plt.plot(mse_history_RMS,label='RMSprop')
plt.plot(mse_history_adam,label='Adam')
plt.plot(mse_history,label='SGD')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("log_opt.pdf", format="pdf")
plt.show()
'''

#----------------------------------------------------------------------------------------------------------
#                    Stochastic GD, MSE vs learning rates after 1000 epochs
#----------------------------------------------------------------------------------------------------------

''' 
learning_rate = np.logspace(-5, -1, 30)
MSE_wo_mom = np.zeros_like(learning_rate)
MSE_w_mom = np.zeros_like(learning_rate)

for i in range(len(learning_rate)):
    MSE_wo_mom[i] = model.stochastic_gradient_descent(learning_rate[i], 0, 0, n_epochs, batch_size, 0, 0)[-1]
    MSE_w_mom[i] = model.stochastic_gradient_descent(learning_rate[i], 0.9, 0, n_epochs, batch_size, 0, 0)[-1]
    print(i)

plt.axhline(y=0.01, color='black', linestyle='dotted', label=r'$\sigma^2 =0.01$', zorder=10)
plt.plot( learning_rate, MSE_wo_mom, label=r'Without momentum, $\gamma  = 0$')
plt.plot(learning_rate, MSE_w_mom, label=r'With momentum, $\gamma  = 0.9$')
plt.xlabel(r'$\eta$')
plt.ylabel('Accuracy')
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig("SGD_MSE_lerningrate.pdf", format="pdf")
plt.show()
'''

#----------------------------------------------------------------------------------------------------------
#                          SGD, MSE, Lambda vs initial learning rate
#----------------------------------------------------------------------------------------------------------

''' 
learning_rate = np.logspace(-5, -1, 9)
lambda_value = np.logspace(-9, 0, 9)
MSE_ridge = np.zeros((len(learning_rate), len(lambda_value)))

for i in range(len(learning_rate)):
    for j in range(len(lambda_value)):
        MSE_ridge[i, j] = model.stochastic_gradient_descent(learning_rate[i],0 , lambda_value[j], n_epochs,  batch_size, 0, 0)[-1]
        print(i, j)

plt.imshow(MSE_ridge, aspect='auto', cmap='rainbow',  interpolation='nearest')
plt.colorbar(label='Accuracy')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\eta$')

lambda_labels = [f"$10^{{{int(np.log10(lam))}}}$" for lam in lambda_value] 
plt.xticks(ticks=np.arange(len(lambda_value)), labels=lambda_labels, rotation=45)

y_ticks_positions = np.linspace(0, len(learning_rate)-1, 5).astype(int)  
learning_rate_labels = [f"$10^{{{int(np.log10(learning_rate[i]))}}}$" for i in y_ticks_positions] 
plt.yticks(ticks=y_ticks_positions, labels=learning_rate_labels)

min_mse = np.max(MSE_ridge)
min_index = np.unravel_index(np.argmax(MSE_ridge), MSE_ridge.shape)
min_x = min_index[1]
min_y = min_index[0]  
plt.plot(min_x, min_y, 'k*', markersize=15, label=f'Best accuracy = {min_mse:.3}')

plt.tight_layout()
plt.legend()
plt.savefig("log_ridge_lambda_lr.pdf", format="pdf")
plt.show()
'''

#----------------------------------------------------------------------------------------------------------
#                SGD, MSE, epochs vs mini batches
#----------------------------------------------------------------------------------------------------------

#''' 
n_epochs = np.array([ 200, 400, 500,  600, 700, 800, 1000])
batch_size = np.array([5, 10, 15, 20, 25, 50, 100])
MSE = np.zeros(( len(n_epochs), len(batch_size)))

for i in range(len(n_epochs)):
    for j in range(len(batch_size)):
        MSE[i, j] = model.stochastic_gradient_descent(learning_rate, 0 , 0, n_epochs[i],  batch_size[j], 0, 0)[-1]
        print(i, j)


plt.imshow(MSE, aspect='auto', cmap='rainbow',  interpolation='nearest')
plt.colorbar(label='Accuracy')
plt.xlabel('Mini batch size')
plt.ylabel('Epochs')

x_ticks_positions = np.arange(len(batch_size))
batch_labels = [str(size) for size in batch_size]
plt.xticks(ticks=x_ticks_positions, labels=batch_labels, rotation=45)

y_ticks_positions = np.arange(len(n_epochs))
epoch_labels = [str(epoch) for epoch in n_epochs]
plt.yticks(ticks=y_ticks_positions, labels=epoch_labels)

min_mse = np.max(MSE)
min_index = np.unravel_index(np.argmax(MSE), MSE.shape)
min_x = min_index[1] 
min_y = min_index[0]  
plt.plot(min_x, min_y, 'k*', markersize=15, label=f'Best accuracy = {min_mse:.3}')

plt.tight_layout()
plt.legend()
plt.savefig("log_epoch_minibatch.pdf", format="pdf")
plt.show()
#'''
