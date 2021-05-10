import sys
import pickle
import numpy as np

fasta_file = "datasets/my_testing_data/test_pdb.fasta"
embeddings_file = "datasets/my_testing_data/test_pdb.protbert-embeddings.pkl"
#distmap_file = "datasets/my_testing_data/icVec.npy" # Only for including structure info.
labels_file = "datasets/my_testing_data/Yterms.pkl"
output_dir = "datasets/my_testing_data/features"

protein_ids = []
sequences = []
file = open(fasta_file, 'r')
for line in file:
    if line[0] == ">": 
        protein_ids.append(line.strip(">\n"))
    elif line != "\n": 
        sequences.append(line.strip())

# print(names)
# print(protein_ids)
# Get embeddings
with open(embeddings_file, 'rb') as fr:
    emb = pickle.load(fr)

# Get labels
with open(labels_file, 'rb') as fr:
    Y = pickle.load(fr)

for i in range(len(protein_ids)):
    feats = {}
    feats['embeddings'] = emb[protein_ids[i]]
    feats['sequence'] = sequences[i]
    feats['labels'] = Y[protein_ids[i]]
    
    with open(output_dir + '/' + protein_ids[i] + '.pkl', 'wb') as fw:
        pickle.dump(feats, fw)

    # Get edges (contact map from distance map)
    # if distmap_file is not None:
    #     distmap = np.load(distmap_file)
    #     contmap_thres = 10.
    #     feats['edges'] = np.where(distmap <= contmap_thres)

# Save features file
#print(protein_feats)

