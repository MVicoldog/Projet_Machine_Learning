from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss


class Svm_Classifier:

    def __init__(self, C):
        self.model = SVC(C=C, probability=True)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def scoreKfold(self, x_train, y_train):
        scores = cross_val_score(
            self.model, x_train, y_train, scoring='accuracy', cv=5)
        return scores

    def global_accuracy(self, x_test, y_test):
        predicted = self.predict(x_test)
        accuracy = (predicted == y_test).mean()
        return accuracy

    def logloss(self, x_test, y_test):
        prediction = self.model.predict_proba(x_test)
        return log_loss(y_test, prediction)

    def getNbSVbyClass(self):
        return self.model.n_support_
