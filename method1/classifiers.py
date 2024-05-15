import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from ast import literal_eval
import networkx as nx
import requests
import numpy as np
import matplotlib.pyplot as plt
from pyvis.network import Network

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralBiclustering
from collections import Counter
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

# from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import r2_score

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class RandomForestForCancer:
    def __init__(self, get_tr_mats=True,csvpath="../BRCA_snv.tsv"):
        self.samples = None
        self.genes = None
        self.sample_gene_matrix = None
        self.gold_df = None
        self.X = None
        self.y = None
        self.X_train,self.y_train,self.X_test,self.y_test = None,None,None,None
        self.X_rci,self.y_rci = None,None
        self.feature_imps = None
        self.dataset = "COAD" if "COAD" in csvpath else "BRCA"

        if(get_tr_mats):
            if("BRCA" in csvpath): self.create_gold_brca()
            else: self.create_gold_coad()
            self.load_data(csvpath=csvpath)
            self.create_training_matrix()
            self.remove_class_imbalance()
            self.train_test_split(type="rci")

    def get_gene_index(self,gene_name):
        return self.genes.tolist().index(gene_name)

    def get_sample_index(self,sample_name):
        return self.samples.tolist().index(sample_name)

    def load_data(self,csvpath="../BRCA_snv.tsv"):
        
        df1 = pd.read_csv(csvpath,delimiter='\t')
        df_heatmap = df1[['Hugo_Symbol','Tumor_Sample_Barcode','SIFT_pred']]
        df_heatmap['SIFT_pred'] = df_heatmap['SIFT_pred'].apply(lambda x: 1 if x=='D' else 0)

        self.samples = df1['Tumor_Sample_Barcode'].unique()
        self.genes = df1['Hugo_Symbol'].unique()

        matrix = np.zeros((len(self.samples),len(self.genes)))
        for i in range(len(df_heatmap)):
            row = df_heatmap.iloc[i]
            gene_index = self.get_gene_index(row['Hugo_Symbol'])
            sample_index = self.get_sample_index(row['Tumor_Sample_Barcode'])
            sift = row['SIFT_pred']
            matrix[sample_index,gene_index] = sift
        
        self.sample_gene_matrix = matrix.copy()
    
    def create_gold_coad(self):
        self.dataset = "COAD"
        df = pd.read_excel("gold.xlsx")
        dfnew = df.iloc[2:].drop(['Supplemental Table S1: List of sample IDs, cancer type and subtype assignments.'],axis=1)
        dfnew= dfnew[dfnew['Unnamed: 2']=="COAD"]
        dfnew=dfnew.rename(columns={"Unnamed: 2":'Disease',"Unnamed: 3":'Subtype',"Unnamed: 1":'Sample'})
        self.gold_df = dfnew.copy()

    def create_gold_brca(self):
        self.dataset = "BRCA"
        df = pd.read_excel("gold.xlsx")
        dfnew = df.iloc[2:].drop(['Supplemental Table S1: List of sample IDs, cancer type and subtype assignments.'],axis=1)
        dfnew= dfnew[dfnew['Unnamed: 2']=="BRCA"]
        dfnew=dfnew.rename(columns={"Unnamed: 2":'Disease',"Unnamed: 3":'Subtype',"Unnamed: 1":'Sample'})
        self.gold_df = dfnew.copy()
    
    def create_training_matrix(self):
        sample_labels = []

        if self.dataset == "BRCA": dict = {'LumA':0, 'Her2':1, 'LumB':2, 'Normal':3, 'Basal':4}
        elif self.dataset == "COAD": dict = {'CIN':0, 'MSI':1, 'GS':2, 'POLE':3}

        for i in range(len(self.sample_gene_matrix)):
            sample_i = self.samples[i]
            print(sample_i[:-13])
            try:
                elt = self.gold_df.loc[self.gold_df['Sample']==sample_i[:-13],'Subtype'].values[0]
                sample_labels.append(dict[elt])
            except:
                sample_labels.append(-1)

        y = np.array(sample_labels)
        X = self.sample_gene_matrix
        self.X = X
        self.y = y
        return X,y

    def train_test_split(self,type="regular",test_size=0.2,print_s=True):
        if(type=="rci"):
            X,y = self.X_rci,self.y_rci
        else:
            X,y = self.X,self.y

        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        # stratified_split.split(X,y)
        for train_index, test_index in stratified_split.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # print(train_index[:50],len(train_index))
            # print(test_index[:50],len(test_index))
            
        if(print_s):
            print("Train data class distribution:")
            print(pd.Series(y_train).value_counts())
            print("\nTest data class distribution:")
            print(pd.Series(y_test).value_counts())
        # return X_train,y_train,X_test,y_test
        self.X_train,self.X_test,self.y_train,self.y_test = X_train,X_test,y_train,y_test
    
    def train_using_gini(self,max_depth=4,min_samples_leaf=5):
        clf_gini = DecisionTreeClassifier(criterion="gini",
                                        random_state=100, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    
        clf_gini.fit(self.X_train, self.y_train)
        return clf_gini

    def train_using_entropy(self,max_depth=4,min_samples_leaf=5):

        # Decision tree with entropy
        clf_entropy = DecisionTreeClassifier(
            criterion="entropy", random_state=100,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf)

        # Performing training
        clf_entropy.fit(self.X_train, self.y_train)
        return clf_entropy

    def get_test_prediction(self,clf_object,print_s=True):
        y_pred = clf_object.predict(self.X_test)
        if(print_s):
            print("Predicted values:")
            print(y_pred)
        return y_pred

    def get_test_accuracy(self,y_test, y_pred,print_s=True):
        if(print_s): print("Confusion Matrix: ",
            confusion_matrix(y_test, y_pred))
        print("Accuracy : ",
            accuracy_score(y_test, y_pred)*100)
        print("R2 Score: ",
            r2_score(y_test,y_pred,multioutput="variance_weighted"))
        if(print_s): print("Report : ",
            classification_report(y_test, y_pred))
        return r2_score(y_test,y_pred,multioutput="variance_weighted"),accuracy_score(y_test, y_pred)*100
        
    def plot_decision_tree(self,clf_object, feature_names, class_names,f=5):
        plt.figure(figsize=(15, 10))
        plot_tree(clf_object, filled=True, feature_names=feature_names, class_names=class_names, rounded=True,fontsize=f)
        plt.show()

    def get_most_important_features(self):
        clf = RandomForestClassifier(max_depth=5, random_state=0)
        clf.fit(self.X_train, self.y_train)
        # y_pred = self.get_test_prediction(clf)
        # self.cal_accuracy(self.y_test,y_pred)

        feat_imps = clf.feature_importances_.tolist()
        tuples_feat_imps = list(enumerate(feat_imps))
        sorted_tuples = sorted(tuples_feat_imps,key=lambda x: x[1],reverse=True)
        top_tuples = [(i[0],i[1]) for i in sorted_tuples if i[1]!=0]
        top_tuples_names = [(self.genes[i[0]],i[1]) for i in top_tuples]
        keys = [i[0] for i in top_tuples_names][:800]
        values = [i[1] for i in top_tuples_names][:800]

        plt.plot(keys,values)

        self.feature_imps = feat_imps
        self.top800 = keys
        return keys

    def remove_class_imbalance(self):
        max = 100 if self.dataset=="BRCA" else 50
        counts = {0:382,1:62,2:154,3:26,4:132}
        count0 = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0

        X_new = []
        y_new = []

        for i in range(len(self.X)):
            y_current = self.y[i]
            X_current = self.X[i]
            
            if(y_current==0 and count0<=max):
                X_new.append(X_current)
                y_new.append(y_current)
                count0+=1
            elif(y_current==1 and count1<=max):
                X_new.append(X_current)
                y_new.append(y_current)
                count1+=1
            elif(y_current==2 and count2<=max):
                X_new.append(X_current)
                y_new.append(y_current)
                count2+=1
            elif(y_current==3 and count3<=max):
                X_new.append(X_current)
                y_new.append(y_current)
                count3+=1
            elif(y_current==4 and count4<=max):
                X_new.append(X_current)
                y_new.append(y_current)
                count4+=1
        
        self.X_rci, self.y_rci = np.array(X_new),np.array(y_new)
        return np.array(X_new),np.array(y_new)