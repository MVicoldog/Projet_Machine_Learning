from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score


class LogReg_Classifier:

    def __init__(self, C):
        """
        Create the model
        :param C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
        """
        self.model = LogisticRegression(C=C, max_iter=500)

    def train(self, x_train, y_train):
        """
        Train model from training dataset
        :param x_train: vectors from training dataset
        :param y_train: classes associated to each vector from x_train
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        """
        predict the value of each vectors in x_test, based on training
        :param x_test: vectors from validating dataset
        """
        return self.model.predict(x_test)

    def scoreKfold(self, x_train, y_train):
        """
        :param x_train: vectors from training dataset
        :param y_train: classes associated to each vector from x_train
        :return: cross-validation score, computed by sklearn.model_selection
        """
        scores = cross_val_score(
            self.model, x_train, y_train, scoring='accuracy', cv=5)
        return scores

    def global_accuracy(self, x_test, y_test):
        """
        :param x_test: vectors from validating dataset
        :param y_test: classes associated to each vector from x_test
        :return: global accuracy on whole validating dataset, based on the model prediction
        """
        predicted = self.predict(x_test)
        accuracy = (predicted == y_test).mean()
        return accuracy

    def logloss(self, x_test, y_test):
        """
        :param x_test: vectors from validating dataset
        :param y_test: classes associated to each vector from x_test
        :return: logloss on whole validating dataset, based on the model prediction
        """
        prediction = self.model.predict_proba(x_test)
        return log_loss(y_test, prediction)

    
    
