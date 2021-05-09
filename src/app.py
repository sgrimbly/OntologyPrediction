"""Template file to develop personal app
WARNINGS: if you run the app locally and don't have a GPU
          you should choose device='cpu'
"""

from typing import Dict, List, Optional
import pprint
import os
import numpy as np
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch import load
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

from model import train, evaluate, load_checkpoint

from biotransformers import BioTransformers
from deepchain.components import DeepChainApp


Score = Dict[str, float]
ScoreList = List[Score]

class App(DeepChainApp):
    """DeepChain App template:

    - Implement score_names() and compute_score() methods.
    - Choose a a transformer available on DeepChain
    - Choose a personal keras/tensorflow model
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self._transformer = BioTransformers(backend="protbert", device=device)
        self._model = Perceptron(input_dim=1024, fc_dim=512, num_classes=256)
        self._loss_fn = torch.nn.MultiLabelSoftMarginLoss()

        self.batch_size = 64 
        self.num_epochs = 100
        self.init_lr = 0.0005 
        self.lr_sched='True'

        self.GO_file = "datasets/for_biodataset_subset/GOs.npy" # Gene Ontology file for classification
        self.data_file = "datasets/for_biodataset_subset/function_prediction.csv"
        self.embeddings_file = "datasets/for_biodataset_subset/embeddings.npy"
        self.labels_file = "datasets/for_biodataset_subset/labels.pkl"

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
        print("here")
        
        
        #self.train_file = "datasets/my_testing_data/train.names" # List of protein_ids for training
        #self.feats_dir = "datasets/my_testing_data/features" # List of features (Seq <-> Embed) for tain/validation
        #self.valid_file = "datasets/my_testing_data/valid.names" # List of protein_ids for validation

        #self.transformer = BioTransformers(backend="protbert", device=device)
        #self.train_data

    @staticmethod
    def score_names() -> List[str]:
        """App Score Names. Must be specified

        Example:
         return ["max_probability", "min_probability"]
        """
        # TODO : Put your own score_names here
        return ["Embedding", "Class"]
        #return ["Loss", "Precision", "ROC AUC", "Min Semantic Distance", "F-score", "True Y", "Sigma Y"]

    def compute_scores(self, sequences) -> ScoreList:
        """Return a list of all proteins score

        Score must be a list of dict:
                - element of list is protein score
                - key of dict are score_names
        """
        # TODO : Fill with you own score function
    
        # Load features: ProteinID <-> Sequence (Input) <-> ProtBERT Embedding
        # Load model if availble, otherwise train model on dataset
        # Evaluate model and get the scores

        # Load checkpoint model and optimizer
        load_checkpoint(self._model)
        embeddings = self._transformer.compute_embeddings(sequences)["cls"]

        sequences = np.array(sequences)
        embeddings = np.array(embeddings)
        
        data = CustomData(x = torch.from_numpy(embeddings))
        scores_list = self._model(data)

        # valid_set = MLPDataset(self.valid_file, self.feats_dir)
        # valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=mlp_collate)

        # score_list = evaluate(
        #     self._device,
        #     self._model, 
        #     self._loss_fn, 
        #     eval_loader = valid_loader, 
        #     icvec = self.GOs, 
        #     nth = 10,
        #     evaluation = True,
        # )
        # print(type(score_list))
        score = dict(zip(self.score_names(), score_list))
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
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=1, collate_fn=mlp_collate)
        train_loader_eval = DataLoader(train_set, batch_size=1, shuffle=False, collate_fn=mlp_collate)

        valid_set = MLPDataset(X_test, y_test)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=mlp_collate)
    
        # Checkpoint and save models during training.
        ckpt_dir = "models_pdb/MLP_E" + '/checkpoint'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        logs_dir = "models_pdb/MLP_E" + '/logs'
        
        # Don't think I need this. Defined in if __name__ == "__main__"
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        
def mlp_collate(batch):
    # Get data, label and length (from a list of arrays)
    #print(batch)
    feats = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    #print(type(feats))
    return CustomData(x = torch.from_numpy(np.array(feats)), y=torch.from_numpy(np.array(labels)))

class CustomData(Data):
    def __init__(self, x=None, mask=None, y=None, **kwargs):
        super(CustomData, self).__init__()

        self.x = x
        self.mask = mask
        self.y = y
        
        for key, item in kwargs.items():
            self[key] = item

class Perceptron(nn.Module):
    def __init__(self, input_dim=1024, fc_dim=512, num_classes=256):
        super(Perceptron, self).__init__()

         # Define fully-connected layers and dropout
        self.layer1 = nn.Linear(input_dim, fc_dim)
        self.drop = nn.Dropout(p=0.4)
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
    # Load the sequences from data.
    seq = ["PKIVILPHQDLCPDGAVLEANSGETILDAALRNGIEIEHACEKSCACTTCHCIVREGFDSLPESSEQEDDMLDKAWGLEPESRLSCQARVTDEDLVVEIPRYTINHARE"]
    app = App("cpu")
    #app.train()
    score_dict = app.compute_scores(seq)
    pprint.pprint(score_dict)


