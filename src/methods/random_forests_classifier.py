from sklearn.ensemble import RandomForestClassifier


class Random_forests_Classifier:

    def __init__(self, n_estimators, max_depth):
        self.model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_valid):
        return self.model.predict(x_valid)

    def global_accuracy(self, x_valid, y_test):
        predicted = self.predict(x_valid)
        accuracy = (predicted == y_test).mean()
        return accuracy
         