import os
import sys
import NNgenomic_v01 as nng
import NNgenomic_multiclase as nngm
import pandas as pd
from utils import create_environment

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'data/')
data = pd.read_csv(DATA_PATH + 'data_pro/intveld_dataprep_tpm_pro_02_corr.csv', index_col=0)
meta_data = pd.read_csv(DATA_PATH + 'data_pro/meta_dataprep_sa_02_corr.csv')

#Binaria
#run_folders = {"model_path": DATA_PATH + "Models/"}
#create_environment(run_folders)
#nn = nng.NeuralNetwork(meta_data, data, run_folders)
#nn.minmax()
#nn.create_model()
#nn.compile_model(weighted_label=True, learning_rate=0.003)
#nn.train_kfold(epochs=30, batch_size=64, k=5)
#nn.evaluate()

#Multiclase
run_folders = {"model_path": DATA_PATH + "ModelsMulti/"}
create_environment(run_folders)
nnm = nngm.NeuralNetwork(meta_data, data, run_folders)
nnm.minmax()

nnm.create_model()
nnm.compile_model(weighted_label=False, learning_rate=0.0001)
nnm.train_kfold(epochs=15, batch_size=64, k=7)
nnm.evaluate()
