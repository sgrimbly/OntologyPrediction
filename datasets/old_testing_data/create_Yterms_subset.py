import itertools
import pickle 

protein_ids = []

fasta_file = "test_pdb.fasta"
file = open(fasta_file, 'r')
for line in file:
    if line[0] == ">": 
        protein_id = line.strip(">\n")
        protein_ids.append(protein_id)

with open("../data_pdb/Yterms.pkl", 'rb') as fr:
    d = pickle.load(fr)

# Select the proteins we are using for test and train
# .toarray() converts sparse matrix into np.array
subset = {k:d[k].toarray() for k in protein_ids if k in d}
#print(subset)
with open("Yterms.pkl", 'wb') as fw:
    pickle.dump(subset, fw)