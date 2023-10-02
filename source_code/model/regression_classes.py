from sklearn.model_selection import KFold
from math import sqrt
import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt 
import warnings
import pickle

class PolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree
    
    def fit_transform(self, X):
        # Transform input features to include polynomial terms
        X_poly = np.column_stack([X ** i for i in range(1, self.degree + 1)])
        return X_poly

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=5)
            
    def __init__(self, regularization, lr, method, momentum, init_theta, num_epochs=500, batch_size=50, cv=kfold):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self.momentum = momentum
        self.init_theta = init_theta

    def mse(self, ytrue, ypred):
        mse = ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
        return mse
    
    def r2(self, ytrue, ypred):
        return 1 - ((((ytrue - ypred) ** 2).sum()) / (((ytrue - ytrue.mean()) ** 2).sum()))
    
    # def fit(self, X_train, y_train):
          
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.infty
        

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            # compute inital theta
            
            if self.init_theta == 'zero':
                self.theta = np.zeros(X_cross_train.shape[1])
            else:
                m = X_cross_train.shape[1]
                sqrt_m = sqrt(m)
                lower, upper = -(1 / sqrt_m), (1 / sqrt_m)
                numbers = np.random.uniform(lower, upper, size=X_cross_train.shape[1])
                scaled = lower + numbers * (upper - lower)
                self.theta = scaled
            
            print('Theta: ', self.theta.shape)
            
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx] 
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    #record dataset
                    mlflow_train_data = mlflow.data.from_numpy(features=X_method_train, targets=y_method_train)
#                     mlflow.log_input(mlflow_train_data, context="training")
                    
                    mlflow_val_data = mlflow.data.from_numpy(features=X_cross_val, targets=y_cross_val)
#                     mlflow.log_input(mlflow_val_data, context="validation")
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
            
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")
            
                    
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        prev_step = 0
        
        step = self.lr * grad
        step += self.momentum * prev_step

        self.theta -= step
        prev_step = step
        
        return self.mse(y, yhat)
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def feature_importance(self):
        return self._coef()
    
class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class NoRegularization:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return 0
        
    def derivation(self, theta): # return 0, since we won't have any regularization.
        return 0
    
class Lasso(LinearRegression):
    
    def __init__(self, method, lr, l, momentum, init_theta):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method, momentum, init_theta)
        
class Ridge(LinearRegression):
    
    def __init__(self, method, lr, l, momentum, init_theta):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, momentum, init_theta)

class Normal(LinearRegression):
    
    def __init__(self, method, lr, init_theta, momentum, l, mlflow_params):
        self.regularization = NoRegularization(l)
        super().__init__(self.regularization, lr, method, momentum, init_theta, mlflow_params)

def load_model2():
    filename2 = '/root/source_code/model/car_prediction2.model'
    loaded_model2 = pickle.load(open(filename2, 'rb'))
    return loaded_model2