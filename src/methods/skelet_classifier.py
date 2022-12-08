from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score


class skelet_classifier:

    def __init__(self) -> None:
        pass

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def scoreKfold(self, x_train, y_train): 
        scores = cross_val_score(self.model, x_train, y_train, scoring='accuracy', cv=10)
        return scores

    def global_accuracy(self, x_test, y_test):
        prediction = self.model.predict(x_test)
        return accuracy_score(y_test, prediction)

    def logloss(self, x_test, y_test):
        prediction = self.model.predict_proba(x_test)
        return log_loss(y_test, prediction)