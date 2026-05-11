from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

class RandomForestModel:
    def train_randomforest(X_train, y_train, X_test, y_test):
        model=RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print("ACCURACY OF RANDOM_FOREST")
        print("Random Forest Training Accuracy :", accuracy_score(y_train, y_pred_train))
        print("Random Forest Testing Accuracy :", accuracy_score(y_test, y_pred_test))

        
        pickle.dump(model,open("Model/random_forest_classifier.pkl","wb"))
        return model