
# Réseau de Neurones Artificiels / Artificial Neural Network




# Part 1 - Préparation des données


# Importion des librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importion des données
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encodage des variables indépendantes catégoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


# Encodage des variables indépendantes catégoriques
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Séparation du jeu de données en un jeu d'entrainement et un de teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Mise à l'échelle des variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  #sans fit pour garder les valeurs issues de l'apprentissage




# Part 2 - Construire le réseau de neurones


# Importation des modules de la librairie Keras 
import keras
from keras.models import Sequential
from keras.layers import Dense


# Initialisation
classifier = Sequential()


# Ajout de la couche d'entrée et de la première couche cachée de neurones
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))


# Ajout de la seconde couche cachée de neurones
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


# Ajout de la couche de sortie de neurones
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compilation du réseau de neurones
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Entrainement du réseau de neurones
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)




# Part 3 - Faire une prédiction et évaluer le modèle


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Faire une matrix de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




# Part 4 - Nouvel prédiction pour un client


# Importation des données
customerdata = pd.read_csv('lambdata_homework.csv')
c = customerdata.iloc[:, 0:10].values


# Encodage des variables indépendantes catégoriques
labelencoder_c_1 = LabelEncoder()
c[:, 1] = labelencoder_c_1.fit_transform(c[:, 1])
labelencoder_c_2 = LabelEncoder()
c[:, 2] = labelencoder_c_2.fit_transform(c[:, 2])


# Encodage des variables indépendantes catégoriques
# et ajout de n_values pour le nombre de valeur par colonnes/variables
onehotencoder = OneHotEncoder(n_values = 3 , categorical_features = [1])
c = onehotencoder.fit_transform(c).toarray()
c = c[:, 1:]


# Mise à l'échelle des variables
c = sc.transform(c)


# Prédiction
y_hat = classifier.predict(c)
y_hat = (y_hat > 0.5)




# Part 5 - Correction
# Ajout de 0.0 au début (au lieu de 0) qui transforme les int en float du tableau
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)




# Part 6 - Evaluation, amélioration et obtimisation du modèle par K-Fold Cross Validation


# Evaluation du modèle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


# Foncton de construction du réseau de neurones
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# K-Fold Cross Validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)


# Entrainement pour obtenir la précision 
# ajouter le paramètre ", n_jobs = -1" pour avoir une répartition sur tout les coeurs l'ordi mais plante 
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)  


# Moyenne
mean = accuracies.mean()


# Ecart type
variance = accuracies.std()




# Amélioration du modèle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Foncton de construction du réseau de neurones avec régulation par abandon qui réduit le surapprentissage au besoin 
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# K-Fold Cross Validation
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)


# Entrainement pour obtenir la précision
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)


# Moyenne
mean = accuracies.mean()


# Ecart type
variance = accuracies.std()




# Obtimisation du modèle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


# Foncton de construction du réseau de neurones
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


# K-Fold Cross Validation
classifier = KerasClassifier(build_fn = build_classifier)


# Définition du dictionnaire
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}


# Création de l'objet de class GridSearchCV
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)


# Lancement de l'entrainement
grid_search = grid_search.fit(X_train, y_train)


# Meilleur configuration 
best_parameters = grid_search.best_params_


# Meillleur précision
best_accuracy = grid_search.best_score_










