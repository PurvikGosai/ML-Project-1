import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle 

class LogisticModel():
    def train_logistic(X_train, y_train, X_test, y_test):
        # print("X_train shape:", X_train.shape)
        # print("y_train shape:", y_train.shape)
        # print("X_test shape:", X_test.shape)
        # print("y_test shape:", y_test.shape)
        

    
        model=LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print("ACCURACY OF Logistic_model")
        print("Logistic_model Training Accuracy :", accuracy_score(y_train, y_pred_train))
        print("Logistic_model Testing Accuracy :", accuracy_score(y_test, y_pred_test))

        ##Saving Model
        pickle.dump(model , open("Model/logistic_model.pkl","wb"))
        ## wb == write in binary mode
        return model