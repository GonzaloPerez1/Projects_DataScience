'''
En el siguiente script podemos encontrar el segundo modelo del trabajo (Artificial Neural Network)
'''
from keras.layers import Dense, Dropout
from keras.models import Sequential
from data_process import separacion_train_test
from aux_functions import exist_model_test
from sklearn.feature_extraction.text import CountVectorizer

def data_vectorized(X_train, X_test):
    vectorizer = CountVectorizer(max_features = 334, stop_words = 'english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer

def ann(X_train_vec):
    ann = Sequential()
    ann.add(Dense(128, input_shape=X_train_vec.shape[1:], activation = 'relu'))
    ann.add(Dropout(0.2))
    ann.add(Dense(50, activation = 'sigmoid'))
    ann.add(Dropout(0.3))
    ann.add(Dense(10, activation = 'sigmoid'))
    ann.add(Dropout(0.4))
    ann.add(Dense(1, activation = 'sigmoid'))

    ann.compile(loss='binary_crossentropy', optimizer='adam',
            metrics=['acc', 'mse', 'mae'])

    return ann

def ann_train(ann_model, X_train_vec, y_train):
    ann_model.fit(X_train_vec.toarray(), y_train, epochs = 10, batch_size = 100, verbose = 0)

    return ann_model

def ann_comprobations(ann_model_train, X_test_vec, y_test):
    prediction = ann_model_train.predict(X_test_vec)
    prediction = list(map(lambda x: 1 if x > 0.5 else 0, prediction))

    accuracy = sum(prediction == y_test) / len(prediction)

    return accuracy

def main():
    #Recogemos los datos de Train y Test
    X_train, X_test, y_train, y_test = separacion_train_test()

    #Vectorizamos los datos
    X_train_vec, X_test_vec, vectorizer = data_vectorized(X_train, X_test)

    #Creamos el modelo y lo entrenamos
    ann_model = ann(X_train_vec)
    ann_model_train = ann_train(ann_model, X_train_vec, y_train) #Modelo a guardar

    #Guardamos el vectorizer y el modelo
    exist_model_test(vectorizer, pickle_name = './model/vector_pickle.p')
    ann_model_train.save('./model/ann_model.h5')

    #Hallamos el accuracy del modelo
    accuracy = ann_comprobations(ann_model_train, X_test_vec, y_test)

    return accuracy