from sklearn.naive_bayes import GaussianNB


class gaussianNB_Classifier:

    def __init__(self, var_smoothing):
        self.model = GaussianNB(var_smoothing = var_smoothing)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def global_accuracy(self, x_test, y_test):
        predicted = self.predict(x_test)
        accuracy = (predicted == y_test).mean()
        return accuracy