import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import dagshub


from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories,save_json



class ModelEvaluation:
    def __init__(self,config : ModelEvaluationConfig):
        self.config=config

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
                distances[i] += np.absolute(data[j] - dataset[i, j])
            
        # Conservation des rangs des k plus proches voisins
        rangs = distances.argsort()
        k_plus_proches_voisins = rangs[:k]
        
        # Calcul de la fréquences des classes parmis les k plus proches voisins
        freq_classes = np.zeros((2,))
        for i in range(k):
            freq_classes[dataset[k_plus_proches_voisins[i]][0]] += 1
        
        # Retour de la classe la plus fréquente de ses voisins
        return int(np.argmax(freq_classes))

    #kppv
    def kppv(self, datas, dataset, k):
        datasClassees = np.zeros(datas.shape[0], dtype=int)
        
        for i in range(datas.shape[0]):
            datasClassees[i] = self.kppv_oneData(datas[i], dataset, k)
        
        return datasClassees

     # Méthode de calcul de l'accuracy
    def accuracy(self, resultats):
        nb_bien_classees = 0
        for i in range(resultats.shape[0]):
            nb_bien_classees += resultats[i,i]
        return nb_bien_classees / resultats.sum()

    # Méthode de calcul de la précision
    def precisions(self, resultats):
        precisions = np.zeros(resultats.shape[0])
        for i in range(resultats.shape[0]):
            precisions[i] = resultats[i, i] / resultats.sum(axis=0)[i]
        return precisions

    # Méthode de calcul du rappel
    def rappels(self, resultats):
        rappels = np.zeros(resultats.shape[0])
        for i in range(resultats.shape[0]):
            rappels[i] = resultats[i, i] / resultats.sum(axis=1)[i]
        return rappels

    # Méthode Naïve Bayes avec retour des mesures d'évaluation
    def kppvV2(self, dataset_test, dataset_apprentissage, k):
        # Table comprenant les résultats de prédictions Références/Hypothèses, initialisée avec des 1 pour éviter les 0 lors des
        # calculs
        table_result = np.ones([2, 2])
        sauv_result = [] # Sauvegarde des résultats brutes pour chaque données
        # Prédiction de chacune des données et affectation dans la table
        resultat = self.kppv(dataset_test, dataset_apprentissage, k)
        sauv_result.append(resultat)
        for i in range(dataset_test.shape[0]):
            table_result[resultat[i], dataset_apprentissage[i][0]] += 1
            
        
        # Calcul des prédictions et des mesures d'évaluation
        return [self.accuracy(table_result), self.precisions(table_result), self.rappels(table_result)]
    
    def predict(self,data):
        spreadsheet = pd.read_csv('artifacts/data/train.csv')
        dataTrain = spreadsheet.to_numpy()
        dataTrain = np.delete(dataTrain, [0, 3, 5, 8, 9, 10, 11], 1)

        self.discretiser_sexe(dataTrain)
        print("Step1 ok")

        dataTests = data.to_numpy()
        dataTest = np.delete(dataTests, [0, 2, 4, 7, 8, 9, 10], 1)
        # Ajout de la colonne des classes
        col = np.zeros((dataTest.shape[0],1), dtype=int)
        dataTest = np.hstack((col, dataTest))

        self.discretiser_sexe(dataTest)

        print("Step2 ok")

        k = int(np.sqrt(dataTrain.shape[0]))

        predictions = self.kppv(dataTest, dataTrain, k)

        
    def eval_metrics(self,actual,pred):
        spreadsheet = pd.read_csv('artifacts/data/train.csv')
        dataTrain = spreadsheet.to_numpy()
        dataTrain = np.delete(dataTrain, [0, 3, 5, 8, 9, 10, 11], 1)

        self.discretiser_sexe(dataTrain)

        data = pd.read_csv('artifacts/data/test.csv')
        dataTests = data.to_numpy()
        dataTest = np.delete(dataTests, [0, 2, 4, 7, 8, 9, 10], 1)
        # Ajout de la colonne des classes
        col = np.zeros((dataTest.shape[0],1), dtype=int)
        dataTest = np.hstack((col, dataTest))

        self.discretiser_sexe(dataTest)

        k = int(np.sqrt(dataTrain.shape[0]))
        acc, prec, rapp = self.kppvV2(dataTrain, dataTrain, k)
        return acc, prec, rapp
    
    def log_into_mlflow(self):
        test_data=pd.read_csv(self.config.test_data_path)

        test_x=test_data.drop([self.config.target_column],axis=1)
        test_y=test_data[[self.config.target_column]]
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
        
        dagshub.init(repo_owner='mohamed.abdhhh', repo_name='titanic-survive', mlflow=True)

        with mlflow.start_run():
            predicted_qualities= self.predict(test_x)
            (acc,prec,rapp)= self.eval_metrics(test_y,predicted_qualities)
            #Saving metrics as local
            prec_survie = prec[0]
            prec_mort = prec[1]
            rapp_survie = rapp[0]
            rapp_mort = rapp[1]
            scores={"acc":acc,"precision survie":prec_survie,"precision rip":prec_mort,"rappel survie":rapp_survie, "rappel rip":rapp_mort}
            save_json(path=Path(self.config.metric_file_name),data=scores)
            
            #mlflow.log_params(self.config.all_params)

            mlflow.log_metric("acc",acc)
            mlflow.log_metric("precision survie",prec_survie)
            mlflow.log_metric("precision rip",prec_mort)
            mlflow.log_metric("rappel survie",rapp_survie)
            mlflow.log_metric("rappel rip",rapp_mort)
            
            
            # Model registry does not work with file store
            #if tracking_url_type_store !="file":

                #mlflow.sklearn.log_model(model,"model",registered_model_name="ElasticnetModel")
            #else:
                #mlflow.sklearn.log_model(model,"model")

