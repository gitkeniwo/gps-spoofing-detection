from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_recall_fscore_support as score
import sklearn.metrics as metrics
import numpy as np
import pandas as pd

def optimize_LocalOutlierFactor(df_train_lof: pd.DataFrame, spoofed_for_test: pd.DataFrame,
              neighbors_arange=(10, 3000, 100)
              ):
    '''
    Gridsearch on neighbors parameter for Optimal Local Outlier Factor.
    '''
    
    neighbors = np.arange(neighbors_arange[0], neighbors_arange[1], neighbors_arange[2])

    neighbor_opt = 0
    fscore = 0

    for neighbor in neighbors:
        lof = LocalOutlierFactor(n_neighbors=neighbor,
                                novelty=True, # novelty detection
                                contamination=0.1, # percentage of outliers
                                n_jobs=-1, # using all processors
                                )   
        lof.fit(df_train_lof.values)
        
        y_pred = lof.predict(spoofed_for_test[['pca-one', 'pca-two', 'pca-three']].values)    
        y_true = spoofed_for_test.label_lof
        conf_matrix = metrics.confusion_matrix(y_true, y_pred)

        precision,recall,temp_fscore,support=score(y_true, y_pred, average='macro', zero_division=0)
        
        if temp_fscore > fscore:
            fscore = temp_fscore
            neighbor_opt = neighbor
            
    return neighbor_opt, fscore

