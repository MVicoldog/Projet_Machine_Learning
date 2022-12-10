from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score


class Random_forests_Classifier:

    def __init__(self, n_estimators, max_depth):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_valid):
        return self.model.predict(x_valid)

    def scoreKfold(self, x_train, y_train):
        scores = cross_val_score(
            self.model, x_train, y_train, scoring='accuracy', cv=5)
        return scores

    def global_accuracy(self, x_valid, y_test):
        predicted = self.predict(x_valid)
        accuracy = (predicted == y_test).mean()
        return accuracy

    def logloss(self, x_test, y_test):
        prediction = self.model.predict_proba(x_test)
        return log_loss(y_test, prediction)
