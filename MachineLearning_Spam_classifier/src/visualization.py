'''
En este script encontramos la visualización de las métricas de cada modelo para ver su rendimiento.
'''

import Naive_Bayes
import ann

def main_info():
    nb_auc, nb_train_score, nb_test_score = Naive_Bayes.main()
    accuracy = ann.main()

    print('\nNAIVE-BAYES')
    print('==========================================')
    print('AUC:',nb_auc )
    print('Train Score:', nb_train_score)
    print('Test Score:', nb_test_score)
    print('\nARTIFICIAL NEURAL NETWORK')
    print('==========================================')
    print('Test Score:', accuracy)