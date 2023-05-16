import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras import backend as K
from utils import plot_result_m
import os
import json

from tensorflow.keras.callbacks import LearningRateScheduler
import math

import warnings
warnings.filterwarnings("ignore")

class NeuralNetwork:
    def __init__(self, meta_data, data, run_folders):
        self.meta_data = meta_data
        self.data = data
        self.run_folders = run_folders
        self.hiperparameters = {}
        #order the clases in order of the meta_data['Type_encoded'] value
        clases_enc = self.meta_data['Type_encoded'].unique()
        indice_ordenado = np.argsort(clases_enc)
        self.classes = self.meta_data['Type'].unique()[indice_ordenado]

        self.train_meta, self.test_meta = train_test_split(self.meta_data, test_size=0.2, stratify=self.meta_data['Type'], random_state=43)
        self.hiperparameters['random_state'] = 43

        class_weight_d = [x for x in self.train_meta['Type_encoded'].value_counts().min() / self.train_meta['Type_encoded'].value_counts()]
        ordered_weights = []
        for t in clases_enc:
            weight_index = list(clases_enc).index(t)
            ordered_weights.append(class_weight_d[weight_index])

        self.class_weight = {t: ordered_weights[i] for i, t in enumerate(clases_enc)}
        print(self.class_weight)
        #self.class_weight = class_weight.compute_class_weight('balanced', np.unique(self.train_meta['Type_encoded']), self.train_meta['Type_encoded'])
        #self.class_weight = {i : self.class_weight[i] for i in range(len(self.class_weight))}

        #self.pos_weight = tf.constant(self.pos_weight, dtype=tf.float32)
        
        self.train_data = self.data[self.train_meta['Patient']]
        self.test_data = self.data[self.test_meta['Patient']]

        self.X_train, self.X_train_se, self.y_train = self.train_data.values.T[:,:-2], self.train_data.values.T[:,-2:], to_categorical(self.train_meta['Type_encoded'])
        self.X_test, self.X_test_se, self.y_test = self.test_data.values.T[:,:-2], self.test_data.values.T[:,-2:], to_categorical(self.test_meta['Type_encoded'])
        self.model = None

    def minmax(self):
        self.hiperparameters['minmax'] = True
        scaler = MinMaxScaler()
        
        self.X_train = scaler.fit_transform(self.train_data.values.T)
        self.X_test = scaler.transform(self.test_data.values.T)

    def create_model(self):
        inputs = Input(shape=(self.X_train.shape[1],))
        x = Dense(2048, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(1024, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)

        outputs = Dense(19, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=outputs)


    def compile_model(self, weighted_label=False, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'], loss=categorical_crossentropy)
        self.hiperparameters['optimizer'] = 'Adam'
        #self.hiperparameters['pos_weight'] = self.class_weight
        if weighted_label == False:
            #creamos un diccionario de unos para que no se aplique el peso
            self.class_weight = {t: 1 for t in self.class_weight.keys()}

    def lr_decay(self, epoch, lr):
        decay_rate = self.learning_rate * 5
        new_lr = lr * (1 / (1 + decay_rate * epoch))
        return new_lr
     

    def train_kfold(self, epochs=100, batch_size=16, k=5):
        csv_logger = CSVLogger(os.path.join(self.run_folders["model_path"], self.run_folders["exp_name"]+'/logs/training.log'), append=True)
        checkpoint = ModelCheckpoint(os.path.join(self.run_folders["model_path"], self.run_folders["exp_name"]+'/weights/NNM_weights.h5'), save_weights_only=True, verbose=1)
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
            y_train, y_val = y[train_index], y[val_index]
            
            # Crear un nuevo modelo y entrenarlo con los datos de entrenamiento
            history = self.model.fit(X_train,
                                 y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=callbacks_list,
                                 class_weight=self.class_weight,
                                 verbose=1)

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
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        self.metrics['test_accuracy'] = test_acc
        self.metrics['test_loss'] = test_loss
        #guarda el diccionario de métricas
        print('--------------------------------------------------------')
        print('Precisión en el conjunto de prueba:', test_acc)
        print('Pérdida en el conjunto de prueba:', test_loss)
        print('--------------------------------------------------------')
        #matriz de confusión 
        y_pred = self.model.predict(self.X_test)
        y_pred_s = np.argmax(y_pred, axis=1)
        y_test_s = np.argmax(self.y_test, axis=1)
        print('Matriz de confusión')
        print(confusion_matrix(y_test_s, y_pred_s))
        print('--------------------------------------------------------')
        print('Reporte de clasificación')
        report = classification_report(y_test_s, y_pred_s, target_names=self.classes, zero_division=0, output_dict=True)
        print(report)
        report_df = pd.DataFrame(report).transpose()
        self.metrics['classification_report'] = report
        #top3 accuracy
        top3_acc = top_k_categorical_accuracy(self.y_test, y_pred, k=3)
        top3_acc = K.mean(top3_acc)
        top3_acc_value = K.get_value(top3_acc)
        self.metrics['top3_accuracy'] = float(top3_acc_value)
        print('Top 3 accuracy: ', top3_acc.numpy())
        print('--------------------------------------------------------')
        top5_acc = top_k_categorical_accuracy(self.y_test, y_pred, k=5)
        top5_acc_k = K.mean(top5_acc)
        top5_acc_value = K.get_value(top5_acc_k)
        self.metrics['top5_accuracy'] = float(top5_acc_value)
        print('Top 5 accuracy: ', top5_acc_k.numpy())
        print('--------------------------------------------------------')
        # TODO: Classification report con el top5
        #print(classification_report(y_test_s, top5_acc, target_names=self.classes, zero_division=0))
        #guarda los resultados
        with open(os.path.join(self.run_folders["model_path"], self.run_folders["exp_name"]+'/logs/metrics.json'), 'w') as fp:
            json.dump(self.metrics, fp)
        with open(os.path.join(self.run_folders["model_path"], self.run_folders["exp_name"]+'/logs/hiperparams.json'), 'w') as fp:
            json.dump(self.hiperparameters, fp)
        with open(os.path.join(self.run_folders["model_path"], self.run_folders["exp_name"]+'/logs/report_df.csv'), 'w') as fp:
            report_df.to_csv(fp)
        plot_result_m(self.model, y_pred_s, y_test_s, self.classes, self.run_folders)
