#import LinearSVClassifier as LSVC


# - Library -
import pandas as pd

# - DataBase Controller -
import sys

import gestion_donnees as gd

# - Controllers -
import controllers.adaBoost_classifier_controller as abcc
import controllers.svm_classifier_controller as svmc
import controllers.LogReg_classifier_controller as lrcc
import controllers.gaussianNB_classifier_controller as gNBcc
import controllers.random_forests_classifier_controller as rfcc
import controllers.neural_network_classifier_controller as nncc

# - Visualizers -
import visualizers.classifierShowdown_Visualizer as cSV
import visualizers.learning_curve as lcV

from sklearn.metrics import log_loss


def main():

    usage = " \n Usage : python .\src\main.py method search_HyperParameters (Classifiers Showdown)\
    \n\n\t method : 1 => Ridge Classifier\
    \n\t method : 2 => Support Vector Classification\
    \n\t method : 3 => GaussianNB Classification\
    \n\t method : 4 => Logistic Regression Classification\
    \n\t method : 5 => Random Forests Classification\
    \n\t method : 6 => Neural Network Classification\
    \n\n\t search_HyperParameters : 0 => Default HyperParameters\
    \n\t search_HyperParameters : 1 => Search HyperParameters\
    \n\n\t Classifiers Showdown : 1 => Make the comparaison between all the model tuned without their hyperparameters\
    \n\t Classifiers Showdown : 2 => Make the comparaison between all the model tuned WITH their hyperparameters"

    # User wants to read usage or error of usage
    if len(sys.argv) == 0 or len(sys.argv) >= 4:
        print(usage)
        return

     # - Gestion Data -
    gestion_donnees = gd.GestionDonnees()
    gestion_donnees.prepocess()
    gestion_donnees.stratifiedSelection()
    x_train, y_train, x_test, y_test = gestion_donnees.x_train, gestion_donnees.y_train, gestion_donnees.x_test, gestion_donnees.y_test

    # Use one classifier
    if len(sys.argv) == 3:
        method = sys.argv[1]
        search_HP = sys.argv[2]
        if search_HP == "0":
            search_HP = False
        elif search_HP == "1":
            search_HP = True

        print("M??thode s??lectionn??e : ")
        if method == "1":
            print("\t- AdaBoost Classifier")
            #controller = rcc.Ridge_Classifier_Controller(search_HP, x_train, y_train)
            controller = abcc.adaBoost_Classifier_Controller(
                search_HP, x_train, y_train, x_test, y_test)
        elif method == "2":
            print("\t- Support Vector Classifier")
            controller = svmc.Svm_Classifier_Controller(
                search_HP, x_train, y_train)
        elif method == "3":
            print("\t- GaussianNB Classifier")
            controller = gNBcc.gaussianNB_Classifier_Controller(
                search_HP, x_train, y_train)
        elif method == "4":
            print("\t- Logistic Regression Classifier")
            controller = lrcc.LogReg_Classifier_Controller(
                search_HP, x_train, y_train, x_test, y_test)
        elif method == "5":
            print("\t- Random Forests Classifier")
            controller = rfcc.Random_Forests_Classifier_Controller(
                search_HP, x_train, y_train, x_test, y_test)
        elif method == "6":
            print("\t- Neural Network Classifier")
            controller = nncc.Neural_Network_Classifier_Controller(
                search_HP, x_train, y_train)
        else:
            print(usage)
            return

        if (controller is None):
            print("\t- Undefined method")
            return
        else:
            classifier = controller.getClassifier()
            if search_HP:
                visualizer = controller.getVisualizer()

                print("Start : Visualisation du score en fonction des param??tres")
                visualizer.Visualise_tuning()
                print("End : Visualisation du score en fonction des param??tre")

        print("Start : Entrainement du mod??le sur les param??tres donn??s")
        classifier.train(x_train, y_train)
        print("End : Entrainement du mod??le sur les param??tres donn??s")

        score = classifier.scoreKfold(x_train, y_train)
        print('kfold score :')
        scores = gd.display_scores(score)
        print(scores)

        if method == "4": 
            print("Start : Visualisation des p??nalties L1 et L2")
            visualizer = controller.getVisualizer()
            visualizer.Visualise_penalty()
            print("End : Visualisation des p??nalties L1 et L2")


        logloss = classifier.logloss(x_train, y_train)
        print('Logloss score sur les donn??es d"entrainement : ', logloss)

        print("Start : Visualisation de l'apprentissage du mod??le")
        title = classifier.__class__.__name__
        lcV.learn_curve.plot_learning_curve(
            classifier.model, title, x_train, y_train, cv=2, scoring="accuracy").show()
        print("End : Visualisation de l'apprentissage du mod??le")


        accuracy = classifier.global_accuracy(x_test, y_test)
        print("Accuracy sur les donn??es de test : ", accuracy)
        
        logloss = classifier.logloss(x_test, y_test)
        print('Logloss score sur les donn??es de test : ', logloss)

    # Compare all classifiers
    if len(sys.argv) == 2:
        showdown = sys.argv[1]
        if showdown == "1":
            search_HP = False
        if showdown == "2":
            search_HP = True

        print("Beginning Confrontation des classifieurs : ")
        results_df = gd.showdown_df()
        classifiers = [
            abcc.adaBoost_Classifier_Controller(
                search_HP, x_train, y_train, x_test, y_test).getClassifier(),
            svmc.Svm_Classifier_Controller(
                search_HP, x_train, y_train).getClassifier(),
            gNBcc.gaussianNB_Classifier_Controller(
                search_HP, x_train, y_train).getClassifier(),
            lrcc.LogReg_Classifier_Controller(
                search_HP, x_train, y_train).getClassifier(),
            rfcc.Random_Forests_Classifier_Controller(
                search_HP, x_train, y_train, x_test, y_test).getClassifier(),
            nncc.Neural_Network_Classifier_Controller(search_HP, x_train, y_train).getClassifier()]

        for clf in classifiers:
            clf.train(x_train, y_train)
            name = clf.__class__.__name__

            # print("="*30)
            # print(name)

            # print('****Results****')

            acc = clf.global_accuracy(x_test, y_test)
            #print("Accuracy: {:.4%}".format(acc))

            # train_predictions = clf.predict_proba(x_test)
            ll = clf.logloss(x_test, y_test)
            # print("Log Loss: {}".format(ll))

            log_entry = gd.showdownPutter(name, acc, ll)

            #results_df = results_df.concat(log_entry)
            results_df = pd.concat([results_df, log_entry])
        print(results_df)
        print("Ending Confrontation des classifieurs : ")
        # cSV.accuracyPlotter(results_df)
        cSV.subPlotter121(results_df)


if __name__ == "__main__":
    main()
