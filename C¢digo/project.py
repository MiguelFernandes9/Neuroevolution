# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:45:48 2019

@author: Raul Barbosa
"""
import pandas as pd
import numpy as np
import FeaturesSelection as FeatureSelection
import Scaling as Scaling
from sklearn.model_selection import train_test_split
import numpy


def func(x):
    if x ==0:
        return 0
    return 1


if __name__ == '__main__':
    #Features Descriptions
    
    dt = pd.read_csv('processed.cleveland.data', sep=',', encoding='ISO-8859-1', engine='python',header=None)
    dt.columns=['Age','Sex','CP','Trestbps','Chol','Fbs','restecg','Thalach','exang','oldpeack','slope','ca','thal','num']
    
    dt = dt.replace('?', np.nan)
    dataset=dt.dropna(how='any')
    dataset['ca'] = dataset['ca'].astype(float)
    dataset['thal'] = dataset['thal'].astype(float)
    dataset['num'] = dataset['num'].apply(func)

    target=dataset['num']
    del dataset['num']
    
    
    '''-------------------------Feature Selection-----------------------------'''
    # Correlation Matriz
    #FeatureSelection.CorrelationMatrizWithHeatMap(dataset, target)
    n_features = 9
    dataset_select_Uni = FeatureSelection.Univariate_Selection(dataset, target, n_features)
    dataset_select = FeatureSelection.Feature_Importance(dataset, target, n_features)
    
    '''-------------------------Divide Dataset into Train and Test-----------------------------'''
    data_train, data_test, target_train, target_test = train_test_split(dataset_select, target, test_size=0.2,
                                                                        random_state=0)
    data_train = data_train.reset_index(drop=True)
    target_train = target_train.reset_index(drop=True)
    
    '''-------------------------Scaling-----------------------------'''
    colunms = list(data_train)
    # data_train_scaled, data_test_scaled = Scaling.Scaling_StandardScaler(data_train, data_test, colunms)
    data_train_scaled, data_test_scaled = Scaling.Scaling_MinMaxScaler(data_train, data_test, colunms)
    #data_train_scaled, data_test_scaled = Scaling.Scaling_RobustScaler(data_train, data_test, colunms)

    Scaling.ScalingComparationScaling(data_train,data_train_scaled)
        
    
    