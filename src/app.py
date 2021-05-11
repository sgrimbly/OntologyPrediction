# Author: St John Grimbly
# Created: 8 May 2021
# Last Updates: 10 May 2021
# Reference: https://github.com/stamakro/GCN-for-Structure-and-Function

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
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from tensorboardX import SummaryWriter

# DeepChain Packages
from biodatasets import load_dataset
from biotransformers import BioTransformers
from deepchain.components import DeepChainApp

# App Classes
# from model import train, evaluate, load_checkpoint

Score = Dict[str, float]
ScoreList = List[Score]

def mlp_collate(self, batch) -> Data:
    """Function for returning single elements (pairs of data) from batch during training."""
    # Get data, label and length (from a list of arrays)
    feats = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    return CustomData(
        x = torch.from_numpy(np.array(feats)), 
        y = torch.from_numpy(np.array(labels))
    )

def train(
    device, 
    net, 
    criterion, 
    learning_rate, 
    lr_sched, 
    num_epochs,
    train_loader, 
    train_loader_eval, 
    valid_loader, 
    icvec, 
    ckpt_dir, 
    logs_dir, 
    evaluate_train = True, 
    save_step = 10):
    """Main function for training the ontology prediction pipeline."""

    # Define logger
    logger = SummaryWriter(logs_dir)

    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Define scheduler for learning rate adjustment
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Load checkpoint model and optimizer
    start_epoch = load_checkpoint(net, optimizer, scheduler, filename=ckpt_dir+'/model_last.pth.tar')

    # Evaluate validation set before start training
    print("[*] Evaluating epoch %d..." % start_epoch)
    avg_valid_loss, avg_valid_avgprec, avg_valid_rocauc, avg_valid_sdmin, avg_valid_fmax, _, _ = evaluate(device, net, criterion, valid_loader, icvec)
    print("--- Average valid loss:                  %.4f" % avg_valid_loss)
    print("--- Average valid avg precision score:   %.4f" % avg_valid_avgprec)
    print("--- Average valid roc auc score:         %.4f" % avg_valid_rocauc)
    print("--- Average valid min semantic distance: %.4f" % avg_valid_sdmin)
    print("--- Average valid max F-score:           %.4f" % avg_valid_fmax)

    # Start training phase
    print("[*] Start training...")

    # Training epochs
    for epoch in range(start_epoch, num_epochs):
        net.train()

        # Print current learning rate
        print("[*] Epoch %d..." % (epoch + 1))

        for param_group in optimizer.param_groups:
            print('--- Current learning rate: ', param_group['lr'])

        for data in train_loader:
            # Get current batch and transfer to device
            data = data.to(device)
            labels = data.y

            with torch.set_grad_enabled(True):  # no need to specify 'requires_grad' in tensors
                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass
                _, outputs = net(data)
                current_loss = criterion(outputs, labels)

                # Backward pass and optimize
                current_loss.backward()
                optimizer.step()

        # Save last model
        state = {
            'epoch': epoch + 1, 
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, ckpt_dir + '/model_last.pth.tar')

        # Save model at epoch
        if (epoch + 1) % save_step == 0:
            print("[*] Saving model epoch %d..." % (epoch + 1))
            torch.save(state, ckpt_dir + '/model_epoch%d.pth.tar' % (epoch + 1))
        
        # Evaluate all training set and validation set at epoch
        print("[*] Evaluating epoch %d..." % (epoch + 1))
        if evaluate_train:
            avg_train_loss, avg_train_avgprec, avg_train_rocauc, avg_train_sdmin, avg_train_fmax, _, _ = evaluate(device, net, criterion, train_loader_eval, icvec)
            print("--- Average train loss:                  %.4f" % avg_train_loss)
            print("--- Average train avg precision score:   %.4f" % avg_train_avgprec)
            print("--- Average train roc auc score:         %.4f" % avg_train_rocauc)
            print("--- Average train min semantic distance: %.4f" % avg_train_sdmin)
            print("--- Average train max F-score:           %.4f" % avg_train_fmax)
            
            logger.add_scalar('train_loss_epoch', avg_train_loss, epoch + 1)
            logger.add_scalar('train_avgprec_epoch', avg_train_avgprec, epoch + 1)
            logger.add_scalar('train_rocauc_epoch', avg_train_rocauc, epoch + 1)
            logger.add_scalar('train_sdmin_epoch', avg_train_sdmin, epoch + 1)
            logger.add_scalar('train_fmax_epoch', avg_train_fmax, epoch + 1)

        avg_valid_loss, avg_valid_avgprec, avg_valid_rocauc, avg_valid_sdmin, avg_valid_fmax, _, _ = evaluate(device, net, criterion, valid_loader, icvec)
        print("--- Average valid loss:                  %.4f" % avg_valid_loss)
        print("--- Average valid avg precision score:   %.4f" % avg_valid_avgprec)
        print("--- Average valid roc auc score:         %.4f" % avg_valid_rocauc)
        print("--- Average valid min semantic distance: %.4f" % avg_valid_sdmin)
        print("--- Average valid max F-score:           %.4f" % avg_valid_fmax)
        
        logger.add_scalar('valid_loss_epoch', avg_valid_loss, epoch + 1)
        logger.add_scalar('valid_avgprec_epoch', avg_valid_avgprec, epoch + 1)
        logger.add_scalar('valid_rocauc_epoch', avg_valid_rocauc, epoch + 1)
        logger.add_scalar('valid_sdmin_epoch', avg_valid_sdmin, epoch + 1)
        logger.add_scalar('valid_fmax_epoch', avg_valid_fmax, epoch + 1)
        
        # LR scheduler on plateau (based on validation loss)
        if lr_sched:
            scheduler.step(avg_valid_loss)

    print("[*] Finish training.")
    return avg_valid_loss, avg_valid_avgprec, avg_valid_rocauc, avg_valid_sdmin, avg_valid_fmax

def evaluate(device, net, criterion, eval_loader, icvec, nth=10, evaluation=False):
    """Evaluate performance of neural network (MLP) on the multi-label classification task."""

    # Eval each sample
    net.eval()
    avg_loss = 0.0
    y_true = []
    y_pred_sigm = []
    with torch.no_grad():   # set all 'requires_grad' to False
        for data in eval_loader:
            # Get current batch and transfer to device
            data = data.to(device)
            labels = data.y

            # Forward pass
            _, outputs = net(data)
            current_loss = criterion(outputs, labels)
            avg_loss += current_loss.item() / len(eval_loader)
            y_true.append(labels.cpu().numpy().squeeze())
            y_pred_sigm.append(torch.sigmoid(outputs).cpu().numpy().squeeze())

        # Calculate evaluation metrics
        y_true = np.vstack(y_true)
        y_pred_sigm = np.vstack(y_pred_sigm)

        # Average precision score
        avg_avgprec = average_precision_score(y_true, y_pred_sigm, average='samples')

        # ROC AUC score
        ii = np.where(np.sum(y_true, 0) > 0)[0]
        avg_rocauc = roc_auc_score(y_true[:, ii], y_pred_sigm[:, ii], average='macro')

        # Minimum semantic distance
        avg_sdmin = smin(y_true, y_pred_sigm, icvec, nrThresholds=nth)

        # Maximum F-score
        avg_fmax = fmax(y_true, y_pred_sigm, nrThresholds=nth)

    return avg_loss, avg_avgprec, avg_rocauc, avg_sdmin, avg_fmax, y_true, y_pred_sigm

def test(device, net, criterion, model_file, test_loader, icvec, save_file=None):
    """Test performance of model on unseen data."""
    # Load pretrained model
    epoch_num = load_checkpoint(net, filename=model_file)
    
    # Evaluate model
    avg_test_loss, avg_test_avgprec, avg_test_rocauc, avg_test_sdmin, avg_test_fmax, y_true, y_pred_sigm = evaluate(device, net, criterion, test_loader, icvec, nth=51)

    # Save predictions
    if save_file is not None:
        pickle.dump({'y_true': y_true, 'y_pred': y_pred_sigm}, open(save_file, 'wb'))

    # Display evaluation metrics
    print("--- Average test loss:                  %.4f" % avg_test_loss)
    print("--- Average test avg precision score:   %.4f" % avg_test_avgprec)
    print("--- Average test roc auc score:         %.4f" % avg_test_rocauc)
    print("--- Average test min semantic distance: %.4f" % avg_test_sdmin)
    print("--- Average test max F-score:           %.4f" % avg_test_fmax)

def load_checkpoint(net, optimizer=None, scheduler=None, filename='model_last.pth.tar'):
    """Custom helper function for loading a checkpointed PyTorch model."""
    start_epoch = 0
    try:
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("\n[*] Loaded checkpoint at epoch %d" % start_epoch)
    except:
        print("[!] No checkpoint found, start epoch 0")

    return start_epoch

def extract(device, net, model_file, names_file, loader, save_file=None):
    """Extract embeddings associated with given model and protein data."""

    # Load pretrained model
    epoch_num = load_checkpoint(net, filename=model_file)

    # Load names file
    names = np.loadtxt(names_file, dtype='str')

    # Extract embeddings
    net.eval()
    embeddings = {}
    with torch.no_grad():   # set all 'requires_grad' to False
        for i, data in enumerate(loader):
            # Get current batch and transfer to device
            data = data.to(device)

            # Forward pass
            emb, _ = net(data)
            embeddings[names[i]] = emb.cpu().numpy().squeeze()

        # Save file
        with open(save_file, 'wb') as f:
            pickle.dump(embeddings, f)

def smin(Ytrue, Ypred, termIC, nrThresholds):
    '''
    Get the minimum normalized semantic distance. This is not really utilised in current
    version of code as we don't make use of structural information. I foresee possible
    use cases in the future and leave this here as a helper function.

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, posterior probabilities (not compressed, in range 0-1).
        termIC: output of ic function above
        nrThresholds: the number of thresholds to check.

    OUTPUT:
        the minimum nsd that was achieved at the evaluated thresholds

    '''

    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ss = np.zeros(thresholds.shape)

    for i, t in enumerate(thresholds):
        ss[i] = normalizedSemanticDistance(Ytrue, (Ypred >=t).astype(int), termIC, avg=True, returnRuMi=False)

    return np.min(ss)

''' helper functions follow '''
def normalizedSemanticDistance(Ytrue, Ypred, termIC, avg=False, returnRuMi = False):
    '''
    Evaluate a set of protein predictions using normalized semantic distance
    value of 0 means perfect predictions, larger values denote worse predictions,

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, predicted binary label ndarray (not compressed). Must have hard predictions (0 or 1, not posterior probabilities)
        termIC: output of ic function above

    OUTPUT:
        depending on returnRuMi and avg. To get the average sd over all proteins in a batch/dataset
        use avg = True and returnRuMi = False
        To get result per protein, use avg = False

    '''

    ru = normalizedRemainingUncertainty(Ytrue, Ypred, termIC, False)
    mi = normalizedMisInformation(Ytrue, Ypred, termIC, False)
    sd = np.sqrt(ru ** 2 + mi ** 2)

    if avg:
        ru = np.mean(ru)
        mi = np.mean(mi)
        sd = np.sqrt(ru ** 2 + mi ** 2)

    if not returnRuMi:
        return sd

    return [ru, mi, sd]

def normalizedRemainingUncertainty(Ytrue, Ypred, termIC, avg=False):
    num =  np.logical_and(Ytrue == 1, Ypred == 0).astype(float).dot(termIC)
    denom =  np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nru = num / denom

    if avg:
        nru = np.mean(nru)

    return nru

def normalizedMisInformation(Ytrue, Ypred, termIC, avg=False):
    num =  np.logical_and(Ytrue == 0, Ypred == 1).astype(float).dot(termIC)
    denom =  np.logical_or(Ytrue == 1, Ypred == 1).astype(float).dot(termIC)
    nmi = num / denom

    if avg:
        nmi = np.mean(nmi)

    return nmi

def fmax(Ytrue, Ypred, nrThresholds):
    thresholds = np.linspace(0.0, 1.0, nrThresholds)
    ff = np.zeros(thresholds.shape)
    pr = np.zeros(thresholds.shape)
    rc = np.zeros(thresholds.shape)

    for i, t in enumerate(thresholds):
        thr = np.round(t, 2)
        pr[i], rc[i], ff[i], _ = precision_recall_fscore_support(Ytrue, (Ypred >=t).astype(int), average='samples')

    return np.max(ff)

def bootstrap(Ytrue, Ypred, ic, nrBootstraps=1000, nrThresholds=51, seed=1002003445):
    '''
    perform bootstrapping (https://en.wikipedia.org/wiki/Bootstrapping)
    to estimate variance over the test set. The following metrics are used:
    protein-centric average precision, protein centric normalized semantic distance, term-centric roc auc

    INPUTS:
        Ytrue : Nproteins x Ngoterms, ground truth binary label ndarray (not compressed)
        Ypred : Nproteins x Ngoterms, posterior probabilities (not compressed, in range 0-1).
        termIC: output of ic function above
        nrBootstraps: the number of bootstraps to perform
        nrThresholds: the number of thresholds to check for calculating smin.

    OUTPUT:
        a dictionary with the metric names as keys (auc, roc, sd) and the bootstrap results as values (nd arrays)
    '''

    np.random.seed(seed)
    seedonia = np.random.randint(low=0, high=4294967295, size=nrBootstraps)

    bootstraps_psd = np.zeros((nrBootstraps,), float)
    bootstraps_pauc = np.zeros((nrBootstraps,), float)
    bootstraps_troc = np.zeros((nrBootstraps,), float)
    bootstraps_pfmax = np.zeros((nrBootstraps,), float)

    for m in range(nrBootstraps):
        [newYtrue, newYpred] = resample(Ytrue, Ypred, random_state=seedonia[m])

        bootstraps_pauc[m] = average_precision_score(newYtrue, newYpred, average='samples')
        bootstraps_psd[m] = smin(newYtrue, newYpred, ic, nrThresholds)

        tokeep = np.where(np.sum(newYtrue, 0) > 0)[0]
        newYtrue = newYtrue[:, tokeep]
        newYpred = newYpred[:, tokeep]

        tokeep = np.where(np.sum(newYtrue, 0) < newYtrue.shape[0])[0]
        newYtrue = newYtrue[:, tokeep]
        newYpred = newYpred[:, tokeep]

        bootstraps_troc[m] = roc_auc_score(newYtrue, newYpred, average='macro')
        bootstraps_pfmax[m] = fmax(newYtrue, newYpred, nrThresholds)

    return {'auc': bootstraps_pauc, 'sd': bootstraps_psd, 'roc': bootstraps_troc, 'fmax': bootstraps_pfmax}

class App(DeepChainApp):
    """Main class containing logic IO and coordinating training and evaluation of model.
    A user can provide an input protein sequence and compute the defined scores classifying
    a protein sequence with some probable functions in the form of a Gene Ontology (GO) 
    identifier. Raw output is a matrix representing scores for each of the possible GOs 
    associated with the dataset the app/model has been trained on.
    
    Please refer to README.md or DESC.md for some more information on what this app does 
    or for references to source material this app is based on."""

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

        # Training locally for testing. App currently makes use of data available in biodatasets.
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
        """Return a list of all proteins score. Required for DeepChain App."""
        # Load checkpoint model and optimizer
        load_checkpoint(net = self._model)
        embeddings = self._transformer.compute_embeddings(sequences)["cls"]
        
        data = CustomData(x = torch.from_numpy(embeddings))
        scores_list = self._model(data)

        score = dict(zip(self.score_names(), scores_list))
        #scores = [{self.score_names(): score} for score in score_list]
        return score

    def train(self) -> None:
        """Coordinates training and validation by collating and calling appropriate helper functions."""
        labels = np.reshape(self.labels, (len(self.labels),-1))
        seq = np.reshape(self.sequences, (len(self.sequences),-1))
        emb = np.reshape(self.embeddings, (len(self.embeddings),-1))
        
        x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(seq, emb, labels, test_size=0.1)
        X_train = (x1_train, x2_train)
        X_test = (x1_test, x2_test)

        train_set = MLPDataset(X_train, y_train)
        train_loader = DataLoader(
            train_set, 
            batch_size = self.batch_size, 
            shuffle = True, 
            num_workers = 1, 
            collate_fn = mlp_collate
        )
        train_loader_eval = DataLoader(
            train_set, 
            batch_size = 1, 
            shuffle = False, 
            collate_fn = mlp_collate
        )

        valid_set = MLPDataset(X_test, y_test)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, collate_fn=mlp_collate)
    
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

class CustomData(Data):
    def __init__(self, x = None, mask = None, y = None, **kwargs) -> None:
        super(CustomData, self).__init__()
        self.x = x
        self.mask = mask
        self.y = y
        
        for key, item in kwargs.items():
            self[key] = item

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



