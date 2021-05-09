
# Gene Ontology Prediction for DeepChain


## Introduction
- I have little to no domain knowledge in protein design or bioinformatics.
- 

## Installation

Follow the instructions at the DeepChainApps GitHub [repo](https://github.com/DeepChainBio/deep-chain-apps) to install your 
environment for creating a DeepChain app. 

I had a lot of trouble working with the deepchain-env and its default install versions. It seemed to install old versions 
of PyTorch, torch-scatter etc. I dealt with this by following the instructions under *Installation via Binaries* on the
[PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#c-cuda-extensions-on-macos).
In short, I installed PyTorch version 1.8.0, and then manually installed torch packages with `pip` by specifying the version I 
need:

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
```

## Protein Function Prediction 
The task I am interested in is taking a [protein sequence](https://en.wikipedia.org/wiki/Protein_primary_structure) and mapping
this to the [protein function](https://en.wikipedia.org/wiki/Protein_function_prediction). This is a notoriously difficult task
since proteins can code for multiple tasks, and different proteins can code for similar functions. This is a dramatic 
oversimplification, but the task is well captured by the work of [The Gene Ontology](https://en.wikipedia.org/wiki/Gene_ontology).

## Method
My plan was to work with a unsupervised prediction method produced by A. Villegas-Morcillo et. al in a paper called 
[Unsupervised protein embeddings outperform hand-crafted sequence and structure features at predicting molecular function](https://academic.oup.com/bioinformatics/article/37/2/162/5892762).
I started by just doing web searches for deep learning methods for function prediction and stumbled across this paper. To ensure I was 
working with reasonable work, and just to get an idea of the field of research available, I mapped out the research [knowledge graph](https://www.connectedpapers.com/main/017a3f8315005d91d90662e95721f39362ca2a7a/Unsupervised-protein-embeddings-outperform-handcrafted-sequence-and-structure-features-at-predicting-molecular-function/graph)
using Connected Papers. This way I got a picture of how this work relates to major works in the field. I briefly read some details in 
major related papers to see what the common approaches to this problem were. During this research, I came across the [BLAST](https://en.wikipedia.org/wiki/BLAST_(biotechnology))
algorithm, which is a more 'brute force' approach to the task of finding similarities between proteins and, by extension, function.

Some other ideas I had were using a CRISPR dataset for cancer genes that had been screened. This led me to the [Cancer Dependency Map](https://depmap.sanger.ac.uk/)
project being run by the Sanger Institute. This project aims to identify dependencies between all cancer cells. My thinking was that we
could take in a gene or protein sequence and predict relation to known cancer dependencies. Perhaps this could help researchers design
cancer therapeutics using InstaDeep's DeepChain platform. 

