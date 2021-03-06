'''
En este scritp se encuentran las funciones que obtienen, limpian y visualizan los datos que usaremos
para el trabajo.
Los correos para identificar spam se van a obtener de una ruta, los datos se encuentran en inglés, esto
es conveniente para poder generalizar más el proyecto debido al gran uso del lenguaje y a que todas las
herramientas están optimizadas al inglés.
'''

import pandas as pd
from sklearn.model_selection import train_test_split

def descargar_dataset():
    dataset = pd.read_csv('./data/spam.csv', encoding= 'latin-1')

    return dataset

def limpiar_dataset():
    dataset = descargar_dataset()

    dataset_limpio = dataset[['Body', 'Label']]

    dataset_limpio = dataset_limpio.dropna()

    return dataset_limpio

def separacion_train_test():
    dataset = limpiar_dataset()

    X = dataset['Body']
    y = dataset['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42)

    return X_train, X_test, y_train, y_test

def visualizar_dataset():
    dataset = limpiar_dataset()
    print(dataset)

def comprobar_datos_train_test():
    X_train, X_test, y_train, y_test = separacion_train_test()

    print('X_train:', X_train.shape)
    print(X_train)
    print('X_test:', X_test.shape)
    print('y_train:', y_train.shape)
    print('y_test:', y_test.shape)