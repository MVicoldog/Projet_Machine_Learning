from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score


class adaBoost_Classifier:

    def __init__(self, base_estimator, n_estimators, learning_rate):
        """
        Create the model
        :param base_estimator: The base estimator from which the boosted ensemble is built.
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
        :param learning_rate: Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier.
        """
        self.model = AdaBoostClassifier(
            base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate)

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
