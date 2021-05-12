# Ontology Prediction 🧬 🎱
This app is a base application that computes the predicted [Molecular Gene Ontology](http://geneontology.org/)
given a protein sequence. It makes use of the `bio-transformers` with a` ESM prot_bert` model as backend to 
compute embeddings for sequences. This work is based on the paper 
[Unsupervised protein embeddings outperform hand-crafted sequence and structure features at predicting molecular function](https://academic.oup.com/bioinformatics/article/37/2/162/5892762).
The data used for training comes from the source code of the paper. I rebundled a portion of this data and 
have made it available in [DeepChain's](https://app.deepchain.bio) open source dataset library, [biodatasets](https://github.com/DeepChainBio/bio-datasets).
Due to limitations of compute the model remains untrained on the full dataset and only a very limited number of
embeddings are pre-computed. Hopefully this will change in the near future.

The default model included is a multi-layer perceptron (MLP) with input size of 
1024, a single hidden layer with 512 fully connected nodes, and an output dimension 256 - the number of GO
classes for the PDB dataset. This model is well captured by the below diagram. Here we see an example input of 
an arbitrary length protein sequence.

![](src/model.png)

The paper this app is based on argues that sequence information with unsupervised embeddings is not aided by
structural information of the protein. Regardless, I think it could be an interesting expansion to include
strucutral information generated by [DeepChain](https://app.deepchain.bio) as additional predictive information 
for the model. This could then be tested against purely sequence embedding based approaches. 

![](src/sequence-function.jpg)

If you'd like to play around and create your own app, check out the DeepChain Apps [GitHub](https://github.com/DeepChainBio/bio-datasets) 
repo for guidance. I am in the process of writing a blog about the process of creating this app. I have some ideas
for other apps that could be interesting and am open to collaboration. Feel free to reach out to [me](https://github.com/sgrimbly)!

## Example
The app is designed to be very easy to use by inputting a sequence to the app. The app will automate
computing an embedding using [biodatasets](https://github.com/DeepChainBio/bio-datasets) ProtBert model. 
This can be run as follows:

```
    seq = [
        "PKIVILPHQDLCPDGAVLEANSGETILDAALRNGIEIEHACEKSCACTTCHCIVREGF \
         DSLPESSEQEDDMLDKAWGLEPESRLSCQARVTDEDLVVEIPRYTINHARE", 
        "PMILGYWNVRGLTHPIRLLLEYTDSSYEEKRYAMGDAPDYDRSQWLNEKFKLGLDFPN \
         LPYLIDGSRKITQSNAIMRYLARKHHLCGETEEERIRVDVLENQAMDTRLQLAMVCYS \
         PDFERKKPEYLEGLPEKMKLYSEFLGKQPWFAGNKITYVDFLVYDVLDQHRIFEPKCL \
         DAFPNLKDFVARFEGLKKISDYMKSGRFLSKPIFAKMAFWNPK"
    ]
    app = App(device = "cpu") # Set to App() for GPU training
    score_dict = app.compute_scores(seq)
    print(score_dict)
```

The app then outputs a numpy array with scores corresponding the the classes - and thus function - it believes
the protein sequence codes for. Since protein sequences can code for multiple functions, and multiple proteins
can code for similar functions, there will likely be a range of scores! An example of output for the above
sequences is:

```
{
    'Class': 
        tensor(
            [[ 0.0256, -0.0338, -0.0292, -0.0446, -0.0345,  0.0050,  0.0508, -0.0102,
            ... 0.0110,  0.0077,  0.0633,  0.0896, -0.0260,  0.0021, -0.1576, -0.0036],
            [ 0.1083, -0.1110, -0.0437, -0.0807,  0.0331, -0.0005,  0.0704, -0.0314,
            ... -0.0331,  0.0576, -0.0238,  0.0333, -0.0444, -0.0256, -0.1072, -0.0553]],
            grad_fn=<AddmmBackward>),

    'Embedding': 
        tensor(
            [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.1248, 0.0276],
            [0.0000, 0.1116, 0.0000,  ..., 0.0000, 0.0000, 0.0356]],
            grad_fn=<MulBackward0>)
}
```

If you take a closer look at the code, you'll notice you can also specify your own dataset by setting the 
`dataset` parameter in the initialisation of the app. This dataset needs to conform to the style of `ontologyprediction`
dataset that I have made available in [biodatasets](https://github.com/DeepChainBio/bio-datasets). Of course, you can
also modify this app by forking my [GitHub repo](https://github.com/sgrimbly/OntologyPrediction) and deploying your own DeepChain app!

## The Dataset
The dataset used during development of this app is largely based on the subset of Protein Data Bank data provided
by the [Unsupervised protein embeddings outperform hand-crafted sequence and structure features at predicting molecular function](https://academic.oup.com/bioinformatics/article/37/2/162/5892762)
paper. The authors lay out the details of how they filtered appropriate sequences for use with their models. 
The main points to consider are that they "selected onsidered proteins with sequence length in the range [40, 1000] 
that had GO annotations in the Molecular Function Ontology (MFO) with evidence codes ’EXP’, ’IDA’, ’IPI’, ’IMP’, ’IGI’, 
’IEP’, ’HTP’, ’HDA’, ’HMP’, ’HGI’, ’HEP’, ’IBA’, ’IBD’, ’IKR’, ’IRD’, ’IC’ and ’TAS’."

# Future Work & Contributing
This app has _loads_ of room for improvement. For one, note that the original paper made use of ELMo embeddings
for training and evaluating their models. They also made use of a wide variety of different models, and used
fairly (empirically) rigorous procedures for testing their models. For example, they performed 5-fold cross 
validation. They also were able to train on the full SwissProt dataset. This dataset can be added in the future,
but calculating the embeddings on a local machine will take a fair amount of compute. Feel free to do this and 
add the embeddings to the open access biodatasets portal 😄 🧬.

This app is open to open source contributions. Please connect with me on the public GitHub repo to discuss
ideas for making this app more generally useful. Some ideas for expansion involve providing more models for
training the multi-label classification task. The original paper using unsupervised embeddings for GO 
classification compared multiple models, including the MLP presented here. Some other interesting models
include GNNs and models that combine sequence information with 3D structural information. 

## Author
[St John Grimbly](https://github.com/sgrimbly)
Research Engineer Intern | InstaDeep 
MSc Applied Mathematics | University of Cape Town

# Tags
## Libraries
- numpy
- pickle
- pandas
- pytorch==1.5
- torch_geometric
- sklearn
- biotransformers
- biodatasets
- deepchain


## Tasks
- transformers
- unsupervised
- multi-task classification

## Embeddings
- ESM

## Datasets / Resources
- [Unsupervised protein embeddings outperform hand-crafted sequence and structure features at predicting molecular function](https://academic.oup.com/bioinformatics/article/37/2/162/5892762)
- [Protein Data Bank](https://www.rcsb.org/) (PDB)
- [Gene Ontology](http://geneontology.org/) (GO) 
