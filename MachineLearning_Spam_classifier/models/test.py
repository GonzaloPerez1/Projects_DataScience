import pickle
import keras

vectorizer = pickle.load(open('vector_pickle.p', 'rb'))
ann_model = keras.models.load_model('ann_model.h5')
naive_model = pickle.load(open('bayes_model.p', 'rb'))

message_test = ['You will win a price in Zalando',
                'Congratulations, you get the job!!',
                'I\'ll kill u']

data = vectorizer.transform(message_test)
prediction_ann = ann_model.predict(data)
prediction_bayes = naive_model.predict(message_test)

print('\nPredicción de la Red Neuronal: ', prediction_ann)
print('\n##########################################################')
print('\nPredicción de Naive-Bayes: ', prediction_bayes)