import pandas as pd
import requests
import numpy as np
from sknetwork.data import from_edge_list
from sknetwork.clustering import Louvain, get_modularity
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class SubnetworkGenerator:
    def __init__(self,list_of_genes=None):
        self.list_of_genes = list_of_genes
        self.gic_master_array = None
        self.exclude_nodes = []

    def load_genes_list(self):
        """
        Returns the list of all genes in the SNV data file
        """
        df1 = pd.read_csv("../BRCA_snv.tsv",delimiter='\t')
        self.list_of_genes = df1['Hugo_Symbol'].unique().tolist()

    def get_density(self,interactions,subnetwork_nodes):
        """
            Input: 
                interactions: List of edges (interactions) with their weights,
                subnetwork_nodes: list of nodes (gene names) in the given subnetwork
            Output: density of the given subnetwork
        """
        score_sum = 0
        for i in range(len(interactions)):
            a = interactions.loc[i].preferredName_A
            b = interactions.loc[i].preferredName_B
            score = float(interactions.loc[i].score)
            if a in subnetwork_nodes and b in subnetwork_nodes:
                score_sum+=score
        
        v = len(subnetwork_nodes)
        density = score_sum/(v*(v-1))
        return density

    def get_interactions(self,protein_list):
        """
        Fetches the interactions between proteins in the given list of proteins from STRING DB
        """
        proteins = '%0d'.join(protein_list)
        species = '9606' # Species number for humans
        url = 'https://string-db.org/api/tsv/network?identifiers=' + proteins + '&species='+species
        # print(url)
        r = requests.get(url)
        lines = r.text.split('\n') # pull the text from the response object and split based on new lines
        data = [l.split('\t') for l in lines] # split each line into its components based on tabs

        # convert to dataframe using the first row as the column names; drop empty, final row
        df = pd.DataFrame(data[1:-1], columns = data[0])

        # dataframe with the preferred names of the two proteins and the score of the interaction
        # print(df.columns)
        interactions = df[['preferredName_A', 'preferredName_B', 'score']]
        return interactions

    def get_gene_index(self,gene_name,list_of_genes):
        """
        Returns the index of a gene in the list of genes
        """
        return list_of_genes.tolist().index(gene_name)

    def get_graph(self,interactions):
        edges = []
        for i in range(len(interactions)):
            edges.append((interactions.iloc[i]['preferredName_A'],interactions.iloc[i]['preferredName_B'],interactions.iloc[i]['score']))
        graph = from_edge_list(edges)
        return graph.names.tolist(),graph.adjacency

    def get_clusters(self,adjacency):
        louvain = Louvain()
        labels = louvain.fit_predict(adjacency)
        labels_unique, counts = np.unique(labels, return_counts=True)
        
        label_counts = {}
        for i,label in enumerate(labels_unique):
            label_counts[label] = counts[i]

        print("Unique cluster labels with counts: ",label_counts)
        print("Modularity:",get_modularity(adjacency, labels))

        return labels

    def get_clustering_dict(self,labels,node_names):
        """
        For the given list of labels corresponding to the list of node names, we output a dictionary of the clusters
        """
        clusters = {}
        for i in range(len(node_names)):
            if(labels[i] in clusters.keys()):
                clusters[labels[i]].append(node_names[i])
            else:
                clusters[labels[i]] = [node_names[i]]

        return clusters

    def draw_graph(self,key_list,clustering_dict,interactions,nodes_to_plot="all"):
        G=nx.Graph(name='Protein Interaction Graph')
        interactions = np.array(interactions)
        for i in range(len(interactions)):
            interaction = interactions[i]
            a = interaction[0] # protein a node
            b = interaction[1] # protein b node
            w = float(interaction[2]) # score as weighted edge where high scores = low weight
            G.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph

        if(nodes_to_plot != "all"):
            # nodes_to_plot = clusters[2]
            subgraph = G.subgraph(nodes_to_plot)
            G = subgraph

        colors = list(mcolors.TABLEAU_COLORS.keys())
        cluster_colors = {i: colors[i % len(colors)] for i in range(len(key_list))}
        node_colors = {}
        for node in G.nodes:
            cluster_found = False
            for cluster_label in key_list:
                if node in clustering_dict[cluster_label]:
                    cluster_found = True
                    node_colors[node] = cluster_colors[cluster_label]
                    break
            if not cluster_found:
                node_colors[node] = 'gray'

        pos = nx.spring_layout(G) # position the nodes using the spring layout
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, with_labels=True, node_color=[node_colors[node] for node in G.nodes], node_size=700, font_size=10,font_weight='bold')
        plt.axis('off')
        plt.show()

    def get_reusable_nodes(self,node_names,labels):
        """
        At most 6 nodes are reused in the next iteration to connect results across iterations
        This function returns the list of reusable nodes, assuming some nodes are excluded.
        """
        clustering_dict = self.get_clustering_dict(labels,node_names)

        reusable = []
        for key in clustering_dict.keys():
            extend = clustering_dict[key][:6]
            for e in extend:
                if e not in self.exclude_nodes:
                    reusable.append(e)
        return reusable


    def greedy_iterative_clustering(self):
        """
        Performs the greedy iterative clustering algorithm to return a master_array containing either the
        cluster IDs of each element in list_of_genes, or a list of cluster IDs for each gene in list_of_genes
        representing the clusters obtained in each iteration. These need to be further analysed to get the
        final set of clusters.
        """
        master_array = []
        for i in range(len(self.list_of_genes)):
            master_array.append([])

        start=0
        indices = list(range(start,start+900))
        start = indices[-1]+1

        ending=False
        add = 0

        while ending==False:
            try:
                genes900 = [self.list_of_genes[i] for i in indices]
            except:
                print("Failed step 1")
                ending = True
                genes900 = []
                for i in indices:
                    if(i<len(self.list_of_genes)):
                        genes900.append(self.list_of_genes[i])

            interactions = self.get_interactions(genes900)
            node_names, adjacency = self.get_graph(interactions)
            cluster_ids = self.get_clusters(adjacency)
            print("Cluster_IDs: ",cluster_ids)

            for i,node in enumerate(node_names):
                try:
                    master_array[self.list_of_genes.index(node)].append(cluster_ids[i]+add)
                    # print("node:",node)
                    # print("List of genes index of node:",list_of_genes.index(node))
                    # print("master array:",master_array)
                    # print(cluster_ids[i],add)
                except:
                    print(node," not in list")
                    self.exclude_nodes.append(node)
            
            reusables = self.get_reusable_nodes(node_names,cluster_ids)

            indices = [self.list_of_genes.index(node) for node in reusables]
            indices.extend(list(range(start,start+900-len(indices))))
            start = indices[-1]+1
            print("Done. Starting at",start," with reusables",reusables)
            add+=900
        
        print("master array:",master_array)
        self.gic_master_array = master_array