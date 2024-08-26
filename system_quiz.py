import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

class ML_System_Regression():
    def __init__(self):
        pass

    def load_data(self):
        path = "D:/Users/Asus/Documents/PYTHON PARA APIS/QUIZ PIPELINE/QUIZ PIPELINE/" 
        dataset = pd.read_csv(path + "iris_dataset.csv",sep=";",decimal=",")
        prueba = pd.read_csv(path + "iris_prueba.csv",sep=";",decimal=",")

        covariables=[x for x in dataset.columns if x not in ["y"] ]
        X = dataset.get(covariables)
        y = dataset["y"]

        X_nuevo = prueba.get(covariables)
        y_nuevo = prueba["y"]

        return X, y, X_nuevo, y_nuevo
    
    def preprocessing_Z(self,X):
        Z = preprocessing.StandardScaler()
        Z.fit(X)
        X_Z= Z.transform(X)
        return Z,X_Z
    
    def trainning_model(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
    
        Z_1, X_train_Z = self.preprocessing_Z(X_train) 
        X_test_Z = Z_1.transform(X_test)
        modelo1 = LogisticRegression(random_state = 123) 
        parametros = {'C': np.arange(0.1,5.1,0.1)}
        grilla1 = GridSearchCV(estimator = modelo1, param_grid = parametros, scoring = make_scorer(accuracy_score), cv = 5, n_jobs = -1)
        grilla1.fit(X_train_Z, y_train)
        y_hat_test = grilla1.predict(X_test_Z)
        u1 = accuracy_score( y_train,  grilla1.predict(X_train_Z))

        y_hat_train = None
    
        Z_2, X_test_Z =self.preprocessing_Z(X_test) 
        X_train_Z = Z_2.transform(X_train)
        modelo2 = LogisticRegression(random_state = 123) 
        grilla2 = GridSearchCV(estimator = modelo2, param_grid = parametros, scoring = make_scorer(accuracy_score), cv = 5, n_jobs = -1)
        grilla2.fit(X_test_Z, y_test)
        y_hat_train = grilla2.predict(X_train_Z)
        u2 = accuracy_score( y_test, y_hat_test)

        if (np.abs(u2 - u1)<10) & (np.abs(grilla1.best_params_['C'] - grilla2.best_params_['C']) < 0.5):
            modelo_completo = LogisticRegression(random_state = 183) 
            parametros = {'C': np.arange(0.1,5.1,0.1)}
            grilla_completa = GridSearchCV(estimator=modelo_completo,param_grid=parametros,cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)
            grilla_completa.fit(X,y)
        else:
            grilla_completa = LogisticRegression(random_state=123)
            grilla_completa.fit(X,y)
        return grilla_completa
    
    def forecast(self,grilla_completa,X_nuevo):
        y_hat_nuevo = grilla_completa.predict(X_nuevo)
        return y_hat_nuevo

    def accuracy(self,y_true,y_hat):
        var = np.abs((y_true-y_hat)/y_true)
        return 100 * np.mean( var<= 0.02 )
    
    def evaluate_model(self,y_nuevo,y_hat_nuevo):
        return self.accuracy(y_nuevo,y_hat_nuevo) 
    
    def ML_Flow_regression(self):
        try:
            X, y, X_nuevo, y_nuevo = self.load_data()
            grilla_completa = self.trainning_model(X,y)
            y_hat_nuevo = self.forecast(grilla_completa,X_nuevo)
            metric = self.evaluate_model(y_nuevo,y_hat_nuevo)
            return {'success':True, 'accuracy':metric }
        except Exception as e:
            return {'succes':False, 'message':str(e)}
              
                           