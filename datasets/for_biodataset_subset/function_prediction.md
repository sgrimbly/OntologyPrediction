# "Protein Function Prediction" dataset

## Dataset Description
Collection of subsets of PDB, and SwissProt based on sequence length (range [40, 1000]) 
and whether proteins have matching Gene Ontology (GO) classifications in the Molecular Function 
Ontology (MFO) with  evidence  codes ’EXP’, ’IDA’, ’IPI’, ’IMP’, ’IGI’, ’IEP’, ’HTP’, ’HDA’, 
’HMP’, ’HGI’,’HEP’, ’IBA’, ’IBD’, ’IKR’, ’IRD’, ’IC’ and ’TAS.

### Dataset Summary

Protein function prediction dataset. ~91k protein sequences. Embeddings available calculated with ProtBert.

Features:
 - Sequence
 - Y


Embeddings:
 - Y embeddings - 1024-dim

Label:
 - binding energy

### Usage
```
from biodatasets import load_dataset

ontology_prediction_dataset = load_dataset("ontology_prediction_pdb")

X, y = ontology_prediction_dataset.to_npy_array(input_names=["sequence","Y"], target_names=["GO"])
embeddings = ontology_prediction_dataset.get_embeddings("sequence", "protbert", "Y")
```

### Supported Tasks
 - GO classification

### Model used to calculate Embeddings
 - ProtBert

### Libraries used to calculate embeddings
 - PyTorch


### Source Data
<!-- [Unsupervised protein embeddings outperform hand-crafted sequence and structure features at predicting molecular function](https://academic.oup.com/bioinformatics/article/37/2/162/5892762#supplementary-data) -->
[DeepChain team](https://deepchain.bio)

### Dataset Curator
[St John Grimbly](https://github.com/sgrimbly)

### Licensing Information
<!-- [Creative Commons Attribution (CC BY 4.0)](https://www.uniprot.org/help/license)  -->
