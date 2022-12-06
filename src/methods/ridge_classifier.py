from sklearn.linear_model import RidgeClassifier

class Ridge_Classifier:

    def __init__(self, alpha):
        self.model = RidgeClassifier(alpha = alpha)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def global_accuracy(self, x_test, y_test):
        predicted = self.predict(x_test)
        accuracy = (predicted == y_test).mean()
        return accuracy