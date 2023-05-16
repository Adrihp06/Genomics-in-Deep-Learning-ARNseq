import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from tensorflow.keras.utils import plot_model
import os
import json



def plot_result(model, fpr, tpr, auc_roc, y_pred, y_test, run_folders):
    plot_model(model, to_file=os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/viz/NNgenomic.png'), show_shapes=True, show_layer_names=True)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (AUC = {:.2f})'.format(auc_roc))
    plt.savefig(run_folders["model_path"] + run_folders["exp_name"]+"/viz/"+"roc_curve.png")
    plt.close()
    
    #ploteamos la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(run_folders["model_path"] + run_folders["exp_name"]+"/viz/"+"confusion_matrix.png")
    plt.close()
    '''
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.axis([0, history.epoch[-1], 0, 1])
    plt.legend(['accuracy', 'loss'], loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(run_folders["model_path"] + run_folders["exp_name"]+"/viz/"+"training_accuracy.png")
    plt.close()
    '''
def plot_result_m(model, y_pred, y_test, classes, run_folders):
    plot_model(model, to_file=os.path.join(run_folders["model_path"], run_folders["exp_name"]+'/viz/NNgenomic.png'), show_shapes=True, show_layer_names=True)

    # ploteamos la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    fig_width = max(6, len(classes) * 0.5)  
    fig_height = max(6, len(classes) * 0.4) 

    # ploteamos la matriz de confusión
    fig, ax = plt.subplots(figsize=(fig_width, fig_height)) 
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    ax.set_xticklabels(classes, rotation=90, ha='right', fontsize=8)

    ax.set_yticklabels(classes, rotation=0, ha='right', fontsize=8)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix')
    plt.tight_layout()
    plt.savefig(run_folders["model_path"] + run_folders["exp_name"]+"/viz/"+"confusion_matrix.png")
    plt.close()
  
    '''
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.axis([0, history.epoch[-1], 0, 1])
    plt.legend(['accuracy', 'loss'], loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(run_folders["model_path"] + run_folders["exp_name"]+"/viz/"+"training_accuracy.png")
    plt.close()
    '''



# Mostrar la figura
plt.show()

def create_environment(run_folders):
    # Creating base folders
    try:
        os.mkdir(run_folders["model_path"])
    except:
        pass
    '''try:
        os.mkdir(os.path.join(run_folders["results_path"], 'log'))
    except:
        pass'''
    # Preparing required I/O paths for each experiment
    if len(os.listdir(run_folders["model_path"])) == 0:
        exp_idx = 1
    else:
        exp_idx = len(os.listdir(run_folders["model_path"])) + 1

    exp_name = "exp_%04d" % exp_idx
    run_folders["exp_name"] = exp_name

    exp_model_folder = run_folders["model_path"] + run_folders["exp_name"] + '/'
    #exp_res_model = run_folders["results_path"] + run_folders["exp_name"] + '/'
    print("Creating experiment folder: ", exp_model_folder)
    try:
        os.mkdir(exp_model_folder)
    except:
        print("Error creating experiment folder")
        pass
    try:
        os.mkdir(exp_res_model)
    except:
        pass
    try:
        os.mkdir(os.path.join(exp_model_folder, 'viz'))
    except:
        pass
    try:
        os.mkdir(os.path.join(exp_model_folder, 'weights'))
    except:
        pass
    try:
        os.mkdir(os.path.join(exp_model_folder, 'logs'))
    except:
        pass

def create_json(hyperparameters, run_folders):
    with open(run_folders["model_path"]+run_folders["exp_name"]+"/hyperparameters.json", 'w') as fp:
        json.dump(hyperparameters, fp)
