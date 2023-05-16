import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import binary_crossentropy
from utils import plot_result
import os
import json

from tensorflow.keras.callbacks import LearningRateScheduler
import math

class NeuralNetwork:
    def __init__(self, meta_data, data, run_folders):
        self.meta_data = meta_data
        self.data = data
        self.run_folders = run_folders
        self.hiperparameters = {}
        self.train_meta, self.test_meta = train_test_split(self.meta_data, test_size=0.2, random_state=10)
        self.hiperparameters['random_state'] = 10
        self.pos_weight = self.train_meta['Classification_Malignant'].value_counts()[0] / self.train_meta['Classification_Malignant'].value_counts()[1]
        #self.pos_weight = tf.constant(self.pos_weight, dtype=tf.float32)
        
        self.train_data = self.data[self.train_meta['Patient']]
        self.test_data = self.data[self.test_meta['Patient']]

        self.X_train, self.X_train_se, self.y_train = self.train_data.values.T[:,:-2], self.train_data.values.T[:,-2:], self.train_meta['Classification_Malignant'].values
        self.X_test, self.X_test_se, self.y_test = self.test_data.values.T[:,:-2], self.test_data.values.T[:,-2:], self.test_meta['Classification_Malignant'].values
        self.model = None

    def minmax(self):
        self.hiperparameters['minmax'] = True
        scaler = MinMaxScaler()
        
        self.X_train = scaler.fit_transform(self.train_data.values.T)
        self.X_test = scaler.transform(self.test_data.values.T)
        ''' 
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.X_train_se = scaler.fit_transform(self.X_train_se)
        self.X_test_se = scaler.transform(self.X_test_se)
        '''       

    def create_model(self):
        inputs = Input(shape=(self.X_train.shape[1],))
        x = Dense(1024, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
 

        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)

        outputs = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=inputs, outputs=outputs)


    def compile_model(self, weighted_label=False, learning_rate=0.01):
        self.learning_rate = learning_rate
        def weighted_cross_entropy_with_logits(logits, targets, pos_weight):
            # logits: salidas de la red neuronal
            # targets: etiquetas verdaderas
            # pos_weight: peso de la clase minoritaria (en este caso, la clase de pacientes sin cáncer)
            targets = tf.cast(targets, tf.float32)
            loss = tf.nn.weighted_cross_entropy_with_logits(targets, logits, pos_weight)
            return tf.reduce_mean(loss)
        self.hiperparameters['learning_rate'] = self.learning_rate
        if weighted_label:
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'], loss=lambda y_true, y_pred: weighted_cross_entropy_with_logits(y_pred, y_true, self.pos_weight)) 
            self.hiperparameters['weighted_label'] = True
            self.hiperparameters['optimizer'] = 'Adam'
            self.hiperparameters['pos_weight'] = self.pos_weight
        else:
            self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'], loss=binary_crossentropy)
            self.hiperparameters['weighted_label'] = False
            self.hiperparameters['optimizer'] = 'Adam'


    def lr_decay(self, epoch, lr):
        decay_rate = self.learning_rate*10
        new_lr = lr * (1 / (1 + decay_rate * epoch))
        return new_lr
    

    def train_kfold(self, epochs=100, batch_size=16, k=5):
        csv_logger = CSVLogger(os.path.join(self.run_folders["model_path"], self.run_folders["exp_name"]+'/logs/training.log'), append=True)
        checkpoint = ModelCheckpoint(os.path.join(self.run_folders["model_path"], self.run_folders["exp_name"]+'/weights/NN_weights.h5'), save_weights_only=True, verbose=1)
        self.hiperparameters['epochs'] = epochs
        self.hiperparameters['batch_size'] = batch_size
        self.hiperparameters['k'] = k

        X = self.X_train
        #X = np.concatenate((self.X_train, self.X_train_se), axis=1)
        y = self.y_train

        lr_scheduler = LearningRateScheduler(self.lr_decay, verbose=1)

        self.metrics = {'train_accuracy': [], 'val_accuracy': [], 'train_loss': [], 'val_loss': []}
        kf = KFold(n_splits=k, shuffle=True, random_state=0)
        # Listas para almacenar los resultados de cada fold
        callbacks_list = [csv_logger, checkpoint, lr_scheduler]
        # Bucle para realizar la validación cruzada k-fold
        for train_index, val_index in kf.split(X):
            # Dividir los datos en entrenamiento y validación
            X_train, X_val = X[train_index], X[val_index]
            #X_train_arn_seq, X_train_categorical = X_train[:, :-2], X_train[:, -2:]
            #X_val_arn_seq, X_val_categorical = X_val[:, :-2], X_val[:, -2:]
            y_train, y_val = y[train_index], y[val_index]
            
            # Crear un nuevo modelo y entrenarlo con los datos de entrenamiento
            
            #self.model.fit([X_train_arn_seq, X_train_categorical], y_train, epochs=epochs, batch_size=batch_size, verbose=1)
            history = self.model.fit(X_train,
                                 y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=callbacks_list,
                                 verbose=1)
            # Evaluar el rendimiento del modelo en los conjuntos de entrenamiento y validación
            #train_accuracy = self.model.evaluate([X_train_arn_seq, X_train_categorical], y_train, verbose=1)[1]
            #val_accuracy = self.model.evaluate([X_val_arn_seq, X_val_categorical], y_val, verbose=1)[1]
            train_loss, train_accuracy = self.model.evaluate(X_train, y_train, verbose=1)
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=1)
            # Almacenar los resultados de este fold
            self.metrics['train_accuracy'].append(train_accuracy)
            self.metrics['val_accuracy'].append(val_accuracy)
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
        print("Resultados de la validación cruzada k-fold:")

        # Calcular y mostrar las métricas promedio de los k folds
        print("----------------------------------------")
        print("Precisión media en el conjunto de entrenamiento: ", np.mean(self.metrics['train_accuracy']))
        print("Precisión media en el conjunto de validación: ", np.mean(self.metrics['val_accuracy']))
        print("Loss media en el conjunto de entrenamiento: ", np.mean(self.metrics['train_loss']))
        print("Loss media en el conjunto de validación: ", np.mean(self.metrics['val_loss']))
        print("----------------------------------------")
        # Guardar el modelo
        return history


       
    def evaluate(self):
        #test_loss, test_acc = self.model.evaluate([self.X_test, self.X_test_se], self.y_test)
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        self.metrics['test_accuracy'] = test_acc
        self.metrics['test_loss'] = test_loss
        #guarda el diccionario de métricas
        print('--------------------------------------------------------')
        print('Precisión en el conjunto de prueba:', test_acc)
        print('Pérdida en el conjunto de prueba:', test_loss)
        print('--------------------------------------------------------')
        #matriz de confusión
        print('Matriz de confusión:')
        y_pred = self.model.predict(self.X_test)
        #curva ROC
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)
        auc_roc = roc_auc_score(self.y_test, y_pred)
        y_pred = np.round(y_pred)
        print(confusion_matrix(self.y_test, y_pred))
        print('--------------------------------------------------------')
        print('Curva ROC: {:.2f}'.format(auc_roc))

        #Calcular la especifidad, sensibilidad y exactitud
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        precision = tp / (tp+fp)
        print('Especificidad: {:.2f}'.format(specificity))
        print('Sensibilidad: {:.2f}'.format(sensitivity))
        print('Precision: {:.2f}'.format(precision))
        print('--------------------------------------------------------')
        #guardar los resultados
        self.metrics['sensitivity'] = sensitivity
        self.metrics['specificity'] = specificity
        self.metrics['precision'] = precision
        with open(os.path.join(self.run_folders["model_path"], self.run_folders["exp_name"]+'/logs/metrics.json'), 'w') as fp:
            json.dump(self.metrics, fp)
        with open(os.path.join(self.run_folders["model_path"], self.run_folders["exp_name"]+'/logs/hiperparams.json'), 'w') as fp:
            json.dump(self.hiperparameters, fp)

        plot_result(self.model, fpr, tpr, auc_roc, y_pred, self.y_test, self.run_folders)
    

