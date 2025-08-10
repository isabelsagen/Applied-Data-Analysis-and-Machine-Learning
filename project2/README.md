# Development and Analysis of a Feed-Forward Neural Network for Classification and Regression
Welcome to our project! This project focuses on developing our own Feed-Forward Neural Network (FFNN) code to tackle both classification and regression problems. Building upon the regression algorithms from Project 1, we aim to implement logistic regression for classification tasks and compare our neural network results against these methods.

We will begin with a regression analysis using a simple second degree polynom, before transitioning to classification with the Wisconsin Breast Cancer dataset. Our implementation will feature plain (GD) and Stochastic Gradient Descent (SGD) and various activation functions, exploring the effects of different learning rates, regularization parameters, and network architectures.

## Code structure 🖥

Our code is structure is split into two, gradient decent and FFNN. The gradient descent part consists of four codes, two classes LinearRegression.py and LogisticRegression.py for the linear and logistic regression class, respectivly. Both codes contains a plain GD and SGD method. There is also an associated "analysis" script for visualisation of linear and logistic regression. The FFNN codes has the following structure; a single class for both regression and classification where you choose either "regression" or "classification" as input task parameter. The class has two associated "analysis" scripts for visualisation. 

To run and visualise the regression we run the desired analysis code. Here the sections tells what we are visualising, and to run remember to remove the comments. 
In the codes for visualisation there are lots of parameters you can change, feel free :)


### Run main.py 🏗📲
To run the code make sure to use Python 3 and have all the used packages intalled. For example if you want to run the analysis_linear_regression.py, make sure to also have LinearRegression.py, and run:

```
python3 analysis_linear_regression.py
```

where the path coincides with where we have our Python script file on our computer. An example path is:

``/Documents/FYS-STK4155/Project2/...``

We hope you have a good time exploring our regressions!
