# Development of differential equation solvers using machine learning

Welcome to our project! This projects focuses on developing three different solvers for differential equations. The first two methods solves the diffusion equation (heat equation), while the last method is a general solver for ODEs using the eigenvalue problem for symmetric matrices. 

## Code structure 🖥
We have grouped our scripts into methods and analysis of the method. This means that you only run the `analysis_xxx.py` scripts for a visualisation of desired method. The methods are forward time centerd space (FTCS), neural network to solve the diffusion equation (NN_pde) and neural network to solve the eigenvalue problem (NN_eigen).

Our code structure is split into two tasks, the first focuses on solving a specific PDE; diffusion equation, using an explicit scheme; the forward time centered space, and a neural network `NN_PDE.py` ; this code trains a neural network to solve the diffusion equation. The last part is a general solver for ODEs; `NN_eigen.py`, this script trains the neural network to compute eigenvalues and eigenvectors of a random symmetric matrix. Each method includes an associated analysis script for visualizing the results. 


### Run the analysis 🏗📲
To run the codes, make sure to use Python 3 and have all the used packages installed. This is TensorFlow, NumPY, Matplotlib and Seaborn. For example if you want to run the analysis_FTCS.py, make sure to also have FTCS.py, and run:

```
python3 analysis_FTCS.py
```

where the path coincides with where we have our Python script file on our computer. An example path is:

``/Documents/FYS-STK4155_P3/...``

We hope you have a good time exploring our PDE solvers!
