## this file will save and train models


import numpy as np
import pandas as pd
from Dataset.preprocessing import DataPreprocessing
from Model.Logistic_model import LogisticModel
from Model.Knn_model import KnnModel
from Model.RandomForestClassifier import RandomForestModel

def main():
    print("LOADING AND PREPROCRDDING DATASET")
    X_train,X_test,y_train,y_test=DataPreprocessing.load_and_preprocess()
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    Lr_model=LogisticModel.train_logistic(X_train, y_train, X_test, y_test)
    Knn_model = KnnModel.train_knn(X_train, y_train, X_test, y_test)
    Random_Forest_Model = RandomForestModel.train_randomforest(X_train, y_train, X_test, y_test)
    
if __name__ == "__main__":
    main()