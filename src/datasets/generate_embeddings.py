import numpy as np
import pickle 

from biotransformers import BioTransformers

#fasta_file = "example/1ax8a.fasta"

fasta_file = "datasets/my_testing_data/test_pdb.fasta"
output_file = "datasets/my_testing_data/test_pdb.protbert-embeddings.pkl"

protein_ids = []
sequences = []
file = open(fasta_file, 'r')
for line in file:
    if line[0] == ">": 
        protein_id = line.strip(">\n")
        protein_ids.append(protein_id)

    elif line != "\n": 
        sequences.append(line.strip())

transformer = BioTransformers(backend="protbert", device="cpu")
emb = np.array(transformer.compute_embeddings(sequences)["cls"])

protbert_embeddings = dict(zip(protein_ids, emb))
with open(output_file, 'wb') as fw:
    pickle.dump(protbert_embeddings, fw)
