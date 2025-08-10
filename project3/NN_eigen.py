import tensorflow as tf
import numpy as np

class NN_eigen:

    #Initialise the Neural Network for solving the eigenvalue problem
    def __init__(self, layers, A, t, max=True):
        self.n = A.shape[0]
        self.A = tf.cast(A, tf.float32)
        self.t = tf.cast(t, tf.float32)
        self.max = max #Wheter to find max or min

        #Define the neural network architecture
        inputs = tf.keras.Input(shape=(1,))
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer["nodes"], activation=layer["activation"])(x)
        outputs = tf.keras.layers.Dense(self.n)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    @tf.function
    #defining the trial function
    def g_trial(self, x, t):
        t = tf.cast(t, tf.float32)
        x = tf.cast(x, tf.float32)
        t = tf.reshape(t, [-1, 1])
        x = tf.reshape(x, [-1, self.n])
        x = tf.repeat(x, tf.shape(t)[0] // tf.shape(x)[0], axis=0)
        model_output = self.model(t)  #Neural network prediction
        exp_term = tf.exp(-t)
        g = exp_term * x + (1 - exp_term) * model_output
        # Normalize g to ensure unit length
        g_norm = tf.sqrt(tf.reduce_sum(tf.square(g), axis=1, keepdims=True))
        g = g / g_norm
        return g

    @tf.function
    #Defining the loss function based on the Reyleigh quotient
    def loss(self):
        v = self.g_trial(self.x, self.t)
        v = tf.reshape(v, [-1, self.n])
        Av = tf.matmul(self.A, v, transpose_b=True)
        vAv = tf.matmul(v, Av)
        vv = tf.reduce_sum(tf.square(v), axis=1, keepdims=True)
        rayleigh_quotient = vAv / vv
        if self.max:
            loss = -tf.reduce_mean(rayleigh_quotient)
        else:
            loss = tf.reduce_mean(rayleigh_quotient)
        return loss

    @tf.function
    #Compute the eigenvalues and the eigenvectors
    def compute_eig(self, x, t):
        v = self.g_trial(x, t)
        v = tf.reshape(v, [-1, self.n])
        Av = tf.matmul(self.A, v, transpose_b=True)
        vAv = tf.matmul(v, Av)
        vv = tf.reduce_sum(tf.square(v), axis=1, keepdims=True)
        eigenvalue = vAv / vv
        eigenvec = v / tf.sqrt(vv)
        return eigenvalue, eigenvec

    @tf.function
    #Compute the gradient of the loss function
    def grad(self):
        with tf.GradientTape() as tape:
            loss = self.loss()
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return loss, gradients
    
    #Set the input data for training
    def set_data(self, x, t):
        self.x0 = tf.cast(x, tf.float32)
        self.t = tf.cast(t, tf.float32)
        
        # Reshape x and t
        x = tf.reshape(self.x0, [1, self.n])
        t = tf.reshape(self.t, [-1, 1])
        
        # Repeat x for each t value
        self.x = tf.repeat(x, tf.shape(t)[0], axis=0)
        self.t = t
   
    #Train the model and predict eigenvalues and eigenvectors
    def predict(self, epochs, x, t, lr):
        self.set_data(x, t)
        eigenvals = np.zeros(epochs)
        eigenvecs = np.zeros([epochs, self.n])
        mse_vals = np.zeros(epochs)
        t_max = tf.reshape(self.t[-1], [-1, 1])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr) #Chosing ADAM optimizer

        for epoch in range(epochs):
            loss, gradients = self.grad()
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            eigenval, eigenvec = self.compute_eig(self.x0, t_max)
            eigenvals[epoch] = tf.reduce_mean(eigenval).numpy()
            eigenvecs[epoch] = tf.reduce_mean(eigenvec, axis=0).numpy()
            mse_vals[epoch] = np.mean((eigenvals[:epoch+1] - eigenval.numpy())**2)
            
            # Decrease learning rate periodically
            if epoch % 1000 == 0 and epoch > 0:
                lr *= 0.1
                optimizer.learning_rate.assign(lr)

        final_mse = mse_vals[-1]
        return eigenvals, eigenvecs, final_mse
        
#Invert the direction of the nn_eigenvec if it is opposite to numpy_eigvec
def align_eigenvector_direction(numpy_eigvec, nn_eigvec):
    # Assuming lengths are equal and both are normalized
    dot_product = np.dot(numpy_eigvec, nn_eigvec)
    if dot_product < 0: # Inverted direction
        return -nn_eigvec
    return nn_eigvec
