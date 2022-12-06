#import LinearSVClassifier as LSVC



# - Library -


# - DataBase Controller -
import sys

import gestion_donnees as gd

# - Controllers -
import controllers.ridge_classifier_controller as rcc
import controllers.svm_classifier_controller as svmc

def main():

    usage= " \n Usage : python .\src\main.py method search_HyperParameters\
    \n\n\t method : 1 => Ridge Classifier\
    \n\t method : 2 => Support Vector Classification\
    \n\n\t search_HyperParameters : 0 => Default HyperParameters\
    \n\t search_HyperParameters : 1 => Search HyperParameters "

    if len(sys.argv) <= 2:
        print(usage)
        return

    method = sys.argv[1]
    search_HP = sys.argv[2]

    if search_HP == "0": search_HP = False
    elif search_HP == "1": search_HP = True
    else:
        print(usage)
        return

    # - Gestion Data -
    train_df, test_df = gd.open_data()
    gestion_donnees = gd.GestionDonnees(train_df=train_df, test_df=test_df)
    gestion_donnees.prepocess()
    gestion_donnees.stratifiedSelection()
    x_train, y_train, x_test, y_test = gestion_donnees.x_train, gestion_donnees.y_train, gestion_donnees.x_test, gestion_donnees.y_test

    print("Selected method : ")
    if method == "1":
        print("\t- Ridge Classifier")
        controller = rcc.Ridge_Classifier_Controller(search_HP, x_train, y_train)
    elif method == "2":
        print("\t- Support Vector Classifier")
        controller = svmc.Svm_Classifier_Controller(search_HP,x_train,y_train)
    else:
        print(usage)
        return

    if (controller is None):
        print("\t- Undefined method")
        return
    else :
        classifier = controller.getClassifier()


    print("Start : Entrainement du modèle sur les paramètres donnés")
    classifier.train(x_train,y_train)
    print("End : Entrainement du modèle sur les paramètres donnés")


    accuracy = classifier.global_accuracy(x_test,y_test)
    print("Accuracy :", accuracy)



if __name__ == "__main__":
    main()