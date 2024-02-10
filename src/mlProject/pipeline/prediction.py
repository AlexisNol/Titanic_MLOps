import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
#from pathlib import Path


class PredictPipelineV2:

    # Méthode de création des intervalles pour la discrétisation du jeu de données
    def discretisation_dataset(dataset, j):
        # Calcul de la moyenne de la colonne
        moy = 0
        for i in range(dataset.shape[0]):
            moy += dataset[i][j]
        moy /= dataset.shape[0]
        
        # Discrétisation des données par rapport à cette valeur
        for i in range(dataset.shape[0]):
            if(dataset[i][j] >= moy):
                dataset[i][j] = 1
            else:
                dataset[i][j] = 0
                
    def discretiser_sexe(self, dataset):
        for i in range(dataset.shape[0]):
            if(dataset[i][2] == 'male'):
                dataset[i][2] = 1
            else:
                dataset[i][2] = 0

    # Méthode K-ppv
    def kppv_oneData(self, data, dataset, k):
        # Calcul des distances entre la donnée de test et l'ensemble des données d'apprentissage selon la formule de la
        # distance de Manhattan
        distances = np.zeros(dataset.shape[0])
        for i in range(dataset.shape[0]):
            for j in range(1, dataset.shape[1]):
                #if not(math.isnan(data[j])) and not(math.isnan(dataset[i, j])):
                distances[i] += np.absolute(data[0][j] - dataset[i, j])
                
        # Conservation des rangs des k plus proches voisins
        rangs = distances.argsort()
        k_plus_proches_voisins = rangs[:k]
        
        # Calcul de la fréquences des classes parmis les k plus proches voisins
        freq_classes = np.zeros((2,))
        for i in range(k):
            freq_classes[dataset[k_plus_proches_voisins[i]][0]] += 1
        
        # Retour de la classe la plus fréquente de ses voisins
        return int(np.argmax(freq_classes))

    def predict(self,data):
        spreadsheet = pd.read_csv('artifacts/data/train.csv')
        dataTrain = spreadsheet.to_numpy()
        dataTrain = np.delete(dataTrain, [0, 3, 5, 8, 9, 10, 11], 1)

        self.discretiser_sexe(dataTrain)
        print("Step1 ok")

        col = np.zeros((data.shape[0],1), dtype=int)
        data_2 = np.hstack((col, data))
        self.discretiser_sexe(data_2)
        data_3 = np.array([int(val) for val in data_2.flatten()], dtype=int)
        data_3 = data_3.reshape(1,5)
        print("Step2 ok")

        k = int(np.sqrt(dataTrain.shape[0]))

        prediction = self.kppv_oneData(data_3, dataTrain, k)

        # Logique conditionnelle pour déterminer le message à afficher
        if prediction == 0:
            prediction_message = "RIP"
        else:
            prediction_message = "Vous avez survécu"

        return prediction_message
        #prediction = self.model.predict(data)
        #return prediction
