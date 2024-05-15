import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class miRNAProcessor:
    def __init__(self,dataset):
        self.dataset = dataset

    def get_miRNA_sample_index(self,samples_mirna,sample):
        return samples_mirna.index(sample)
    
    def get_combined_multiomic(self,data_matrix,transf_matrix,samples):
        matrix_multiomic = np.zeros((len(data_matrix),len(data_matrix[0])+len(transf_matrix[0])))
        for i in range(len(samples)):
            s_name = samples[i][:16]
            try:
                s_id_mirna = self.get_miRNA_sample_index(self.samples_mirna,s_name)
            except:
                pass

            matrix_multiomic[i] = data_matrix[i].tolist() + transf_matrix[s_id_mirna].tolist()
        return matrix_multiomic

    def load_miRNA(self,csvpath="BRCA_miRNA.tsv"):
        if(self.dataset=="COAD"): csvpath="COAD_miRNA.tsv"
        miRNA = pd.read_csv(csvpath,delimiter="\t")
        miRNA_filtered = miRNA[['Sample ID','miRNA_ID','reads_per_million_miRNA_mapped']]
        self.samples_mirna=miRNA_filtered['Sample ID'].unique().tolist()
        miRNA_IDs = miRNA_filtered['miRNA_ID'].unique().tolist()

        matrix_miRNA = np.zeros((len(self.samples_mirna),len(miRNA_IDs)))

        k = 0
        miRNA_filt_temp = miRNA_filtered.copy()

        for i in tqdm(range(len(self.samples_mirna))):
            for j in range(len(miRNA_IDs)):
                samp = self.samples_mirna[i]
                mirna = miRNA_IDs[j]
                
                elt = miRNA_filt_temp.iloc[k]
                # print(elt,samp,mirna)
                # print()

                if elt['Sample ID']==samp and elt['miRNA_ID']==mirna:
                    matrix_miRNA[i,j] = elt['reads_per_million_miRNA_mapped']
                    # print(k)
                    
                k+=1

        scaler = StandardScaler()
        scaler.fit(matrix_miRNA)
        print(scaler.mean_,scaler.var_)
        transf_matrix = scaler.transform(matrix_miRNA)
        sns.heatmap(transf_matrix)

        return transf_matrix

