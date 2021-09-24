'''
En este script encotraremos las funciones auxiliares utilizadas para propósitos generales
'''
import pickle
import os

def pickle_save(archivo, pickle_name):
    pickle_data = (archivo)
    pickle.dump(pickle_data, open(pickle_name, 'wb'))

def exist_model_test(archivo, pickle_name):
    if os.path.exists(pickle_name):
        os.remove(pickle_name)
        pickle_save(archivo, pickle_name)
    else:
        pickle_save(archivo, pickle_name)