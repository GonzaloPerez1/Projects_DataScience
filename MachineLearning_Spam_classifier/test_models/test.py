'''
En este script encontramos las funciones necesarias para testear nuestros modelos (Naive-Nayes y Artificial Neural Network),
lo haremos tanto con mensajes propios como con un dataset descargado de kaggle (https://www.kaggle.com/uciml/sms-spam-collection-dataset)
'''

import pickle
import keras
import pandas as pd

def import_models():
    vectorizer = pickle.load(open('vector_pickle.p', 'rb'))
    ann_model = keras.models.load_model('ann_model.h5')
    naive_model = pickle.load(open('bayes_model.p', 'rb'))

    return vectorizer, ann_model, naive_model

def conversion_target(x):
    if x == 'ham':
        x = 0
    else:
        x = 1

    return x

def data_test():
    #Soluciones (1,0,0,0,1,1,0,0)
    message_test = ['You will win a price in Zalando',
                    'Congratulations, you get the job!!',
                    'I\'ll kill u',
                    'Nah I don\'t think he goes to usf, he lives around here though',
                    'SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575',
                    'URGENT! You have won a 1 week FREE membership in our �100,000 Prize Jackpot! Txt the word: CLAIM to 25001',
                    'I HAVE A DATE ON SUNDAY WITH WILL!!',
                    'Fine if that��s the way u feel. That��s the way its gota b']

    dataset = pd.read_csv('./data_test/spam_messages/spam.csv', sep=',', encoding='latin-1')

    message_test_1 = dataset['v2'][:15]
    solutions = dataset['v1'][:15]
    solutions = solutions.apply(conversion_target)

    return message_test, message_test_1, solutions

def ann(ann_model, vectorizer, message):
    data_ann = vectorizer.transform(message)

    prediction_ann = ann_model.predict(data_ann)
    prediction_ann = list(map(lambda x: 1 if x > 0.5 else 0, prediction_ann))

    return prediction_ann

def naive_bayes(naive_model, message):
    prediction_bayes = naive_model.predict(message)

    return prediction_bayes

def visualization_predictions(prediction_ann, prediction_bayes, prediction_ann_1, prediction_bayes_1, solutions):
    print('\n##########################################################')
    print('Datos Propios:')
    print('\tPredicción de la Red Neuronal:', prediction_ann)
    print('\tPredicción de Naive-Bayes:', prediction_bayes)
    print('##########################################################')
    print('Datos Kaggle:')
    print('\tPredicción de la Red Neuronal:', prediction_ann_1)
    print('\tPredicción de Naive-Bayes:', prediction_bayes_1)
    print('##########################################################')
    print('Soluciones:')
    print('\tSoluciones propias: [1,0,0,0,1,1,0,0]')
    print('\tSoluciones Kaggle:', list(solutions))
    print('##########################################################')

def main():
    vectorizer, ann_model, naive_model = import_models()

    message_test, message_test_1, solutions = data_test()

    prediction_ann = ann(ann_model, vectorizer, message_test)
    prediction_bayes = naive_bayes(naive_model, message_test)

    prediction_ann_1 = ann(ann_model, vectorizer, message_test_1)
    prediction_bayes_1 = naive_bayes(naive_model, message_test_1)

    visualization_predictions(prediction_ann, prediction_bayes, prediction_ann_1, prediction_bayes_1, solutions)

if __name__ == '__main__':
    main()