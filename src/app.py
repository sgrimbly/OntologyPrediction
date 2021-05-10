# Author: St John Grimbly
# Created: 8 May 2021
# Last Updates: 10 May 2021

# Common Python Libraries
from typing import Dict, List, Optional
import pprint
import os
import numpy as np
import pickle
import pandas as pd

# (Deep) ML Libraries
import torch
import torch.nn as nn
from torch import load
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# DeepChain Packages
from biodatasets import load_dataset
from biotransformers import BioTransformers
from deepchain.components import DeepChainApp

# App Classes
from model import train, evaluate, load_checkpoint

Score = Dict[str, float]
ScoreList = List[Score]

class App(DeepChainApp):
    def __init__(self, dataset: str = "ontologyprediction-test", device: str = "cuda:0") -> None:
        self._device = device
        self._dataset = load_dataset(dataset)
        self._transformer = BioTransformers(backend="protbert", device=device)

        self._model = Perceptron(input_dim=1024, fc_dim=512, num_classes=256)
        self._loss_fn = torch.nn.MultiLabelSoftMarginLoss()

        self.batch_size = 64 
        self.num_epochs = 100
        self.init_lr = 0.0005 
        self.lr_sched='True'

        self.GO_file = str(self._dataset.path) + '/GOs.npy' # Gene Ontology file for classification
        self.data_file = str(self._dataset.path) + '/function_prediction.csv'
        self.embeddings_file = str(self._dataset.path) + '/embeddings.npy'
        self.labels_file = str(self._dataset.path) + '/labels.pkl' # Pickle (Rick) Dict of labels

        # self.GO_file = "datasets/for_biodataset_subset/GOs.npy" # Gene Ontology file for classification
        # self.data_file = "datasets/for_biodataset_subset/function_prediction.csv"
        # self.embeddings_file = "datasets/for_biodataset_subset/embeddings.npy"
        # self.labels_file = "datasets/for_biodataset_subset/labels.pkl"

        self.GOs = np.load(self.GO_file).astype(np.float32)
        self.data = pd.read_csv(self.data_file)

        self.protein_ids = self.data["protein_id"].to_numpy()
        self.sequences = self.data["sequence"].to_numpy()
        self.embeddings = np.load(self.embeddings_file).astype(np.float32)

        d = pickle.load(open(self.labels_file, 'rb'))
        self.labels = []
        for i in range(len(self.protein_ids)):
            self.labels.append(d[self.protein_ids[i]])
        self.labels = np.array(self.labels)
        
    @staticmethod
    def score_names() -> List[str]:
        """App Score Names. Required for DeepChain App."""

        return ["Embedding", "Class"]
        #return ["Loss", "Precision", "ROC AUC", "Min Semantic Distance", "F-score", "True Y", "Sigma Y"]

    def compute_scores(self, sequences) -> ScoreList:
        """Return a list of all proteins score. Required for DeepChain App.

        Score must be a list of dict:
                - element of list is protein score
                - key of dict are score_names
        """
        # Load checkpoint model and optimizer
        load_checkpoint(self._model)
        embeddings = self._transformer.compute_embeddings(sequences)["cls"]
        
        data = CustomData(x = torch.from_numpy(embeddings))
        scores_list = self._model(data)

        score = dict(zip(self.score_names(), scores_list))
        #scores = [{self.score_names(): score} for score in score_list]
        return score

    def train(self) -> None:
        labels = np.reshape(self.labels, (len(self.labels),-1))
        seq = np.reshape(self.sequences, (len(self.sequences),-1))
        emb = np.reshape(self.embeddings, (len(self.embeddings),-1))
        
        x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(seq, emb, labels, test_size=0.1)
        X_train = (x1_train, x2_train)
        X_test = (x1_test, x2_test)

        train_set = MLPDataset(X_train, y_train)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=1, collate_fn=Utils.mlp_collate)
        train_loader_eval = DataLoader(train_set, batch_size=1, shuffle=False, collate_fn=Utils.mlp_collate)

        valid_set = MLPDataset(X_test, y_test)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=Utils.mlp_collate)
    
        # Checkpoint and save models during training.
        ckpt_dir = "models_pdb/MLP_E" + '/checkpoint'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        logs_dir = "models_pdb/MLP_E" + '/logs'
        
        # Training and validation
        score_list = train(
            device = self._device, 
            net = self._model, 
            criterion = self._loss_fn,
            learning_rate = self.init_lr, 
            lr_sched = self.lr_sched, 
            num_epochs = self.num_epochs,
            train_loader = train_loader, 
            train_loader_eval = train_loader_eval, 
            valid_loader = valid_loader,
            icvec = self.GOs, 
            ckpt_dir = ckpt_dir, 
            logs_dir = logs_dir
        )

        return score_list

class Utils:        
    def mlp_collate(self, batch) -> Data:
        # Get data, label and length (from a list of arrays)
        feats = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        return CustomData(
            x = torch.from_numpy(np.array(feats)), 
            y = torch.from_numpy(np.array(labels))
        )

class CustomData(Data):
    def __init__(self, x = None, mask = None, y = None, **kwargs) -> None:
        super(CustomData, self).__init__()
        self.x = x
        self.mask = mask
        self.y = y
        
        for key, item in kwargs.items():
            self[key] = item

class Perceptron(nn.Module):
    """Model used for predicting the (Molecular) Gene Ontology (GO).
    NOTE The input dimension is determined by the shape of the 
    embeddings given to the model. The number of classes is defined 
    by the dataset we are using. In the origin paper (Villegas-Morcillo et al.)
    256 classes are used for the PDB data.
    """
    def __init__(self, input_dim = 1024, fc_dim = 512, num_classes = 256) -> None:
        super(Perceptron, self).__init__()

         # Define fully-connected layers and dropout
        self.layer1 = nn.Linear(input_dim, fc_dim)
        self.drop = nn.Dropout(p = 0.4)
        self.layer2 = nn.Linear(fc_dim, num_classes)

    def forward(self, data):
        x = data.x

        # Compute fully-connected part and apply dropout
        x = nn.functional.relu(self.layer1(x))
        x = self.drop(x)
        embedding = x
        output = self.layer2(x)   # sigmoid in loss function

        return embedding, output

class MLPDataset(Dataset):
    def __init__(self, X, y):
        # Initialize data
        self.sequences = X[0]
        self.embeddings = X[1]
        self.y = y

    def __len__(self):
        # Get total number of samples
        return len(self.y)

    def __getitem__(self, index):
        return self.embeddings[index].T, self.y[index]

if __name__ == "__main__":
    # Example sequences.
    seq = [
        "PKIVILPHQDLCPDGAVLEANSGETILDAALRNGIEIEHACEKSCACTTCHCIVREGF \
         DSLPESSEQEDDMLDKAWGLEPESRLSCQARVTDEDLVVEIPRYTINHARE", 
         
        "PMILGYWNVRGLTHPIRLLLEYTDSSYEEKRYAMGDAPDYDRSQWLNEKFKLGLDFPN \
         LPYLIDGSRKITQSNAIMRYLARKHHLCGETEEERIRVDVLENQAMDTRLQLAMVCYS \
         PDFERKKPEYLEGLPEKMKLYSEFLGKQPWFAGNKITYVDFLVYDVLDQHRIFEPKCL \
         DAFPNLKDFVARFEGLKKISDYMKSGRFLSKPIFAKMAFWNPK"
    ]
    app = App(device = "cpu")
    #app.train()
    score_dict = app.compute_scores(seq)
    pprint.pprint(score_dict)


