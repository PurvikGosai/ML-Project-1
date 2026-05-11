from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

class KnnModel:
    def train_knn(X_train, y_train, X_test, y_test):
        model=KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print("ACCURACY OF Knn")
        print("Knn Training Accuracy :", accuracy_score(y_train, y_pred_train))
        print("Knn Testing Accuracy :", accuracy_score(y_test, y_pred_test))

        pickle.dump(model , open("Model/knn_model.pkl","wb"))
        return model
