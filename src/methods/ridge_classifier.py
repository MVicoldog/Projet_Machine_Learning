from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score


class Ridge_Classifier:

    def __init__(self, alpha):
        self.model = RidgeClassifier(alpha = alpha)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def scoreKfold(self, x_train, y_train):
        scores = cross_val_score(self.model, x_train, y_train, scoring='accuracy', cv=5)
        return scores

    def global_accuracy(self, x_test, y_test):
        predicted = self.predict(x_test)
        accuracy = (predicted == y_test).mean()
        return accuracy

    def logloss(self, x_test, y_test):
        return 0