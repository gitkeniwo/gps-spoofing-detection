from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
import numpy as np
import math
import pandas as pd

def optimize_OneClassSVM(X, n):
    """
    optimize_OneClassSV：： Optimize the hyperparameters of the One-Class SVM model
    
    Explanation
    -------
    The function searches for the optimal hyperparameters of the One-Class SVM model. 
    The hyperparameters are optimized by minimizing the difference between the predicted
    
    Parameters
    ----------
    X : np.array
        The input data
    n : int
        The number of hyperparameters to search for

    Returns
    -------
    nu_opt, gamma_opt = optimize_OneClassSVM(X, n)
    """
    
    print('Searching for optimal hyperparameters...')
    
    nu = np.linspace(start=5e-2, stop=0.9, num=n)
    gamma = np.linspace(start=1e-5, stop=1e-1, num=n)
    
    opt_diff = 1.0 # difference between the predicted and expected error rate
    opt_nu = None # optimal nu
    opt_gamma = None # optimal gamma
    
    for i in range(len(nu)):
        
        for j in range(len(gamma)):
            
            classifier = OneClassSVM(kernel="rbf", nu=nu[i], gamma=gamma[j])
            
            classifier.fit(X)
            
            label = classifier.predict(X)
            
            p = 1 - float(sum(label == 1.0)) / len(label)
            
            diff = math.fabs(p - nu[i]) # difference between the predicted and expected error rate
            
            if diff < opt_diff: # update the optimal hyperparameters
                opt_diff = diff
                opt_nu = nu[i]
                opt_gamma = gamma[j]
                
    print(f"Found: nu = {opt_nu}, gamma = {opt_gamma}")
    
    return opt_nu, opt_gamma


# Grid search for the optimal hyperparameters
from sklearn.model_selection import GridSearchCV

N = 20
nu = np.linspace(start=1e-4, stop=0.9, num=N)
gamma = np.linspace(start=1e-6, stop=1e-3, num=N)
param_grid = {
    'kernel' : ['rbf', 'linear', 'poly', 'sigmoid'], 
    'gamma' : gamma, 
    'nu': nu
    }

SCORES = ['precision', 'recall']

def gridsearch_ocsvm(x_train, y_train, params = param_grid, scores=SCORES):

    for score in SCORES:
        
        clf = GridSearchCV(OneClassSVM(), 
                           param_grid, 
                           cv=10, 
                           scoring='%s_macro' % score, 
                           return_train_score=True)

        clf.fit(x_train, y_train)

        resultDf = pd.DataFrame(clf.cv_results_)
        print(resultDf[["mean_test_score", "std_test_score", "params"]].sort_values(by=["mean_test_score"], ascending=False).head())

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        
        return clf.best_params_