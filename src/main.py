#import LinearSVClassifier as LSVC



# - Library -


# - DataBase Controller -
import sys

import gestion_donnees as gd

# - Controllers -
import controllers.ridge_classifier_controller as rcc
import controllers.svm_classifier_controller as svmc
import controllers.LogReg_classifier_controller as lrcc
import controllers.gaussianNB_classifier_controller as gNBcc
import controllers.random_forests_classifier_controller as rfcc
import controllers.neural_network_classifier_controller as nncc

def main():

    usage= " \n Usage : python .\src\main.py method search_HyperParameters\
    \n\n\t method : 1 => Ridge Classifier\
    \n\t method : 2 => Support Vector Classification\
    \n\t method : 3 => GaussianNB Classification\
    \n\t method : 4 => Logistic Regression Classification\
    \n\t method : 5 => Random Forests Classification\
    \n\t method : 6 => Neural Network Classification\
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
    elif method == "3":
        print("\t- GaussianNB Classifier")
        controller = gNBcc.gaussianNB_Classifier_Controller(search_HP,x_train,y_train)
    elif method == "4":
        print("\t- Logistic Regression Classifier")
        controller = lrcc.LogReg_Classifier_Controller(search_HP,x_train,y_train)
    elif method == "5":
        print("\t- Random Forests Classifier")
        controller = rfcc.Random_Forests_Classifier_Controller(search_HP,x_train,y_train)
    elif method == "6":
        print("\t- Neural Network Classifier")
        controller = nncc.Neural_Network_Classifier_Controller(search_HP,x_train,y_train)
    else:
        print(usage)
        return

    if (controller is None):
        print("\t- Undefined method")
        return
    else :
        classifier = controller.getClassifier()
        visualizer = controller.getVisualizer()
    
    print("Start : Visualisation du score en fonction des paramètres")
    visualizer.Visualise()
    print("End : Entrainement du modèle sur les paramètres donnés")

    print("Start : Entrainement du modèle sur les paramètres donnés")
    classifier.train(x_train,y_train)
    print("End : Entrainement du modèle sur les paramètres donnés")


    accuracy = classifier.global_accuracy(x_test,y_test)
    print("Accuracy :", accuracy)



if __name__ == "__main__":
    main()