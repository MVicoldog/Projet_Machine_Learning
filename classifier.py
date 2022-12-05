from abc import ABC, abstractmethod

class Classifier(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def entrainement(self,x_train,t_train):
        pass

    @abstractmethod
    def prediction(self,x):
        pass

    @staticmethod
    @abstractmethod
    def erreur(self,t,prediction):
        """
        Retourne l'erreur de classification
        """
        pass

    @abstractmethod
    def affichage(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        pass

    @abstractmethod
    def parametre(self):
        """
        Retourne les paramètres du modèle
        """
        pass


