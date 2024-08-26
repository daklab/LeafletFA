import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import os
from tqdm import tqdm
import torch 
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pdb
import scipy.sparse as sp

sys.path.append('/gpfs/commons/home/kisaev/Leaflet-private/src/clustering/')
import load_cluster_data as llc 

# write function that takes in Cluster name 
def check_SS_cluster(final_data, junc_info, cluster_name):
    
    # this cluster can also be used to label intron clusters as exon inclusion/exclusion event 
    juncs_c = junc_info[junc_info["Cluster"] == cluster_name]
    
    # keep only rows where either start or end appear twice
    s = np.array(juncs_c[juncs_c.duplicated(subset=['start'])].start.unique())
    e = np.array(juncs_c[juncs_c.duplicated(subset=['end'])].end.unique())
    juncs_c = juncs_c[(juncs_c["start"].isin(s)) | (juncs_c["end"].isin(e))]

    # if num rows in juncs_c is 3 then return cluster name 
    if len(juncs_c == 3):
        # confirm cluster has non zero counts in both cell types 
        num_celltypes = len(final_data[final_data.Cluster == cluster_name].cell_type.unique())
        if num_celltypes > 1:
            # confirm that each junction has non zero counts in both cell types
            counts_juncs = final_data[final_data.Cluster == cluster_name].junction_id.value_counts()
            counts_juncs = counts_juncs[counts_juncs > 1]
            num_cells=len(counts_juncs.index.unique())
            if num_cells == 3:
                return cluster_name
    else:
        pass

def simulate_junc_counts(cluster_counts, junc_info, cell_types=None, psi_prior_shape1=0.5, psi_prior_shape2=0.5):
    
    """Simulate junc counts while keeping the cluster counts of observed data. 
    
    Args: 
        cluster_counts: scipy coo_matrix. 
        cell_types: pandas Categorical series of pre-defined cell types to use for simulations 
        psi_prior_shape1: float.
        psi_prior_shape2: float.
    
    Returns:
        sim_junc_counts: scipy coo_matrix. 
        cell_type_labels: numpy array of cell type labels. 
        cell_type_psi: numpy array of cell type specific PSI values.
    """
    
    N, P = cluster_counts.shape  # number of cells, number of junctions
     
    # use real cell types labels to represent intron clusters being higher/lower in specific cell types 
    print("Using pre-defined cell types!")
    cell_type_labels = cell_types.cat.codes.to_numpy()
    K = len(cell_types.cat.categories)  # number of cell types
    print("The number of cell types is:", K)
    print("The number of cells is:", N)
    print("The number of junctions is:", P)
    
    # number of intron clusters 
    num_clusters = len(junc_info.Cluster.unique())

    # label clusters as positive or negative by sampling 
    cluster_labels = np.random.choice([0, 1], size=num_clusters)

    # make a mapping of Cluster ID to cluster_labels
    cluster_labels_dict = dict(zip(junc_info.Cluster.unique(), cluster_labels))

    # initiate empty dataframe cell_type_psi_df to which we will append the simulated PSI values for each junction in each cell type
    cell_type_psi_df = pd.DataFrame()

    for clust in tqdm(junc_info.Cluster.unique()):
        clust_label = cluster_labels_dict[clust]
        # get cluster label
        # get junctions in cluster and order them by start and end 
        juncs_c = junc_info[junc_info["Cluster"] == clust]
        # order juncs_c by start and end
        juncs_c = juncs_c.sort_values(by=['start', 'end'])
        # assign J1, J2, and J3 to junctions where J1+J2 correspond to exon inclusion and J3 corresponds to exon skipping
        juncs_c["junction"] = ["J1", "J3", "J2"] 
        num_juncs = len(juncs_c)    
        if clust_label == 0: 
            # sample PSI values for each junction in each cell type via pre-defined beta distributions
            probs = torch.distributions.beta.Beta(psi_prior_shape1, psi_prior_shape2).sample([num_juncs, K]) 
            # get J3 prob 
            probs[1,] = probs[1,1]
            probs[0,] = (1-probs[1,1])/2
            probs[2,] = (1-probs[1,1])/2
            # convert probs to dataframe 
            probs_df = pd.DataFrame(probs.cpu().numpy())  # Adjusted line
            # probs_df = pd.DataFrame(probs.numpy())
            # add junction_id_index column to probs_df
            probs_df["new_junction_id_index"] = juncs_c["new_junction_id_index"].values
            probs_df["sample_label"] = "negative"
            probs_df["Cluster"] = juncs_c["Cluster"].values[0]
            # appent probs_df to cell_type_psi_df
            cell_type_psi_df = pd.concat([cell_type_psi_df, probs_df])
        elif clust_label == 1:
            probs = torch.distributions.beta.Beta(psi_prior_shape1, psi_prior_shape2).sample([num_juncs, K]) 
            # get J3 prob
            J3_prob = probs[1,]
            probs[0,] = (1-J3_prob)/2
            probs[2,] = (1-J3_prob)/2
            probs_df = pd.DataFrame(probs.cpu().numpy())  # Adjusted line
            # probs_df = pd.DataFrame(probs.numpy())
            probs_df["new_junction_id_index"] = juncs_c["new_junction_id_index"].values
            probs_df["sample_label"] = "positive"
            probs_df["Cluster"] = juncs_c["Cluster"].values[0]
            # use pd.concat to append probs_df to cell_type_psi_df
            cell_type_psi_df = pd.concat([cell_type_psi_df, probs_df])

    cell_type_psi_df = cell_type_psi_df.sort_values(by=['new_junction_id_index'])

    # keep just the first K columns of cell_type_psi_df
    # specify which columns to keep (K columns)
    cols_keep = cell_type_psi_df.columns[0:K]
    print("The columns to keep are:", cols_keep)
    cell_type_psi = torch.tensor(cell_type_psi_df[cols_keep].to_numpy())
    # cell_type_psi = torch.tensor(cell_type_psi_df[[0,1]].to_numpy()) #should specify K columns insted of "0,1"
    print("Done simulating PSI!")

    # use real cluster counts to simulate junc counts with binomial distribution
    sim_junc_counts = cluster_counts.copy() 
    sim_junc_counts.data = torch.distributions.binomial.Binomial(
        total_count=torch.tensor(cluster_counts.data), 
        probs=cell_type_psi[
            cluster_counts.col,  # junction index
            cell_type_labels[cluster_counts.row]  # cell index 
        ]
    ).sample().cpu().numpy()  # Adjusted line to add .cpu() before .numpy()

    print("Done simulating junc counts!")
    
    return sim_junc_counts, cell_type_labels, cell_type_psi, cell_type_psi_df

# use sim_dat to get cluster level PSI for each cluster using J1+J2/J3 
def get_cluster_PSI(cluster, sim_data, junc_info):
    # cell clust counts 
    clust_only_counts = sim_data[sim_data["Cluster"] == cluster]

    # junction label info (J1, J2, J3)
    clust_dat = junc_info[junc_info["Cluster"] == cluster]
    clust_dat.sort_values(by = ["start", "end"], inplace = True)
    clust_dat["junc_label"] = ["J1", "J3", "J2"]
    clust_dat = clust_dat[["Cluster", "junc_label", "new_junction_id_index"]]
    # rename new_junction_id_index column to junction_id_index 
    clust_dat.rename(columns = {"new_junction_id_index": "junction_id_index"}, inplace = True)
    clust_only_counts = sim_data[sim_data["Cluster"] == cluster]
    #merge clust_dat with clust_only_counts 
    clust_only_counts = clust_only_counts.merge(clust_dat, on = ["Cluster", "junction_id_index"])
    clust_only_counts = clust_only_counts[["cell_id_index", "Cluster", "junction_id_index", "junc_count", "cluster_count", "junc_label"]]
    clust_only_counts.sort_values(by = ["cell_id_index", "junction_id_index"], inplace = True)
    
    clust_only_counts_mat = clust_only_counts.pivot(index = "junction_id_index", columns = "cell_id_index", values = "junc_count")
    cols_names = list(clust_only_counts_mat)
    clust_only_counts_mat["junc_label"] = clust_only_counts[["junction_id_index", "junc_label"]].drop_duplicates()["junc_label"].values
    # reorder rows using junc_label
    clust_only_counts_mat.sort_values(by = ["junc_label"], inplace = True)
    # for each column calculate J1+J2/J1+J2+J3
    # for each cell in each column sum first two rows and divide by sum of all three rows
    clust_cells_psi = []
    for i in cols_names:
        col_index = clust_only_counts_mat.columns.get_loc(i)
        psi_value = clust_only_counts_mat.iloc[2,col_index].sum() / clust_only_counts_mat.iloc[:,col_index].sum()
        # save cell index (i), cluster and psi value
        clust_cells_psi.append([i, cluster, psi_value])

    # convert clust_cells_psi to dataframe
    clust_cells_psi = pd.DataFrame(clust_cells_psi, columns = ["cell_id_index", "Cluster", "cluster_psi"])
    return(clust_cells_psi)

def quick_clust_plot(clust, simple_data, num_cols=3, plot_states_arb=True):
    
    simple_data_junc = simple_data[simple_data["Cluster"] == clust]
    # make violin plot with jitter 
    print(simple_data_junc.cell_type.value_counts())
    sample_label = simple_data_junc.sample_label.unique()[0]
    
    # if plot_states_arb = True then instead of real cell types make dummy variable and use that for cell type
    if plot_states_arb:
        # get unique values in cell type and mapping to a number 
        cell_types = simple_data_junc.cell_type.unique()
        cell_type_map = dict(zip(cell_types, range(len(cell_types))))
        simple_data_junc["cell_type"] = simple_data_junc["cell_type"].map(cell_type_map)
        #make sure new values in cell_Type are string
        simple_data_junc["cell_type"] = simple_data_junc["cell_type"].astype(str)
    
    plt.figuresize=(6, 6)

    # choose three distrinct colours to use for junction_id_index hue 
    colors = sns.color_palette("husl", num_cols)

    # use colors in violinplot
    sns.violinplot(data = simple_data_junc, x = "junc_ratio", y = "cell_type", hue="junction_id_index", palette=colors)

    # make xlim -1 to 1.1
    plt.xlim(-0.2, 1.2)
    # add sample_label to title 
    plt.title(sample_label + " label for cluster:" + str(clust), fontsize=16)
    # set x axis label to "Junction Usage Ratio (PSI)"
    plt.xlabel("Junction Usage Ratio (PSI)", fontsize=20)
    plt.ylabel("Cell Type Group", fontsize=20)
    # increase x and y tick label size to 14
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # put legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.show()

def simulate_and_prepare_data(input_files_folder, K, float_type, max_intron_count=1000):
    """Load, filter, and simulate data, returning tensors for the model."""
    
    # Load real data
    print(f"Load real data from {input_files_folder}...")

    final_data, coo_counts_sparse, coo_cluster_sparse, cell_ids_conversion, junction_ids_conversion = llc.load_cluster_data(
        input_folder=input_files_folder, max_intron_count=max_intron_count, remove_singletons=True, has_genes="yes") 

    print(final_data.columns)
    
    # add cluster to final_data 
    # final_data = final_data.merge(junction_ids_conversion, on=["junction_id_index", "junction_id", "Cluster"], how="left")

    # get indices (maybe don't need this actually)
    indices = (final_data.cell_id_index, final_data.junction_id_index)
    junc_counts = sp.coo_matrix((final_data.junc_count, indices))
    cluster_counts = sp.coo_matrix((final_data.Cluster_Counts, indices))

    # Sanity check that counts are saved in correct indices in sparse matrices
    ind_random = np.random.randint(0, len(final_data))
    print(final_data.iloc[ind_random])
    print(junc_counts.toarray()[final_data.iloc[ind_random].cell_id_index, final_data.iloc[ind_random].junction_id_index])
    print(cluster_counts.toarray()[final_data.iloc[ind_random].cell_id_index, final_data.iloc[ind_random].junction_id_index])

    # SS are shared between end of J1 and start of J2 and end of J2 and start of J3
    junc_info = junction_ids_conversion[["junction_id", "Cluster", "junction_id_index"]].drop_duplicates()

    # get number of junctions in each cluster first 
    cluster_junc_counts = junc_info.groupby(["Cluster"]).agg({"junction_id": "count"}).reset_index()
    clusts_keep = cluster_junc_counts[cluster_junc_counts["junction_id"] == 3 ]
    junc_info = junc_info[junc_info["Cluster"].isin(clusts_keep["Cluster"])]

    # break up junction_id column in junc_info into chr, start and end 
    junc_info["chr"] = junc_info["junction_id"].str.split("_").str[0]
    junc_info["start"] = junc_info["junction_id"].str.split("_").str[1]
    junc_info["end"] = junc_info["junction_id"].str.split("_").str[2]
    print(len(junc_info["Cluster"].unique()))

    # run function on all clusters to find simple exon skipping events 
    clusters_SS = []

    print(f"Annotating Clusters to find those with exon skipping events!")
    for cluster in tqdm(junc_info["Cluster"].unique()): # this is very slow
        clusters_SS.append(check_SS_cluster(final_data, junc_info, cluster))

    # keep only entries in clusters_SS that are not None 
    clusters_SS = [x for x in clusters_SS if x is not None]
    print(len(clusters_SS))
    
    # get indices of junctions in clusters_SS (original indices before filtering)
    junc_ind_keep = junction_ids_conversion[junction_ids_conversion["Cluster"].isin(clusters_SS)]["junction_id_index"]
    final_data = final_data[final_data.junction_id_index.isin(junc_ind_keep)] #using original junction id index

    print(f"Filter junction_ids file to only include junctions in exon skipping clusters")
    junction_ids_conversion = junction_ids_conversion[junction_ids_conversion["junction_id_index"].isin(junc_ind_keep)]
    # reset index of junction_ids_conversion and make a new column new_junction_id_index
    junction_ids_conversion = junction_ids_conversion.reset_index(drop=True)
    # re-order junction_ids_conversion junction_id_index
    junction_ids_conversion = junction_ids_conversion.sort_values(by=['junction_id_index'])
    junction_ids_conversion["new_junction_id_index"] = junction_ids_conversion.index

    # re-order the remaining junctions and subset the counts matrices
    final_data = final_data.merge(junction_ids_conversion, on=["junction_id_index", "Cluster", "junction_id"])

    # where is new_junction_id_index coming from here? 
    final_data.sort_values(by = ["new_junction_id_index"], inplace = True)
    final_data.head()

    to_keep = final_data["junction_id_index"].unique()   # use original junction indices to filter out the count matrices 
    junc_counts_sub = junc_counts.tocsr()[:,to_keep].tocoo()
    cluster_counts_sub = cluster_counts.tocsr()[:,to_keep].tocoo()

    # Sanity check that counts are saved in correct indices in sparse matrices
    ind_random = np.random.randint(0, len(final_data))
    print(final_data.iloc[ind_random])
    print(junc_counts_sub.toarray()[final_data.iloc[ind_random].cell_id_index, final_data.iloc[ind_random].new_junction_id_index])
    print(cluster_counts_sub.toarray()[final_data.iloc[ind_random].cell_id_index, final_data.iloc[ind_random].new_junction_id_index])

    print(f"Let's simulate some data!")
    # update junc_info to only include junctions in clusters_SS
    junc_info = junc_info[junc_info["Cluster"].isin(clusters_SS)]
    junc_info = junc_info.reset_index(drop=True)
    junc_info["new_junction_id_index"] = junc_info.index

    print(f"The number of unique junctions and clusters included in the simulation data is: ")
    print(len(junc_info.junction_id.unique()))
    print(len(junc_info.Cluster.unique()))

    # TO-DO change this step so can work with any number of "cell types"
    cell_ids_conversion["cell_type"] = np.random.choice([1,2], size=len(cell_ids_conversion))

    simulated_counts, cell_types, cell_type_psi, cluster_labels = simulate_junc_counts(cluster_counts_sub, junc_info, cell_types=cell_ids_conversion.cell_type.astype('category'))
    
    # Check outcome of cluster_labels
    print(cluster_labels)

    # save simulated counts, cell types and psi values
    sim_juncs_counts = simulated_counts
    cell_type_psi_df = cluster_labels

    # Get variance in simulated psi values across all simulated cell types 
    print(cell_type_psi_df.head())
    cell_type_psi_df["difference"] = cell_type_psi_df[0] - cell_type_psi_df[1]
    cell_type_psi_df["difference"] = np.abs(cell_type_psi_df["difference"])

    # TO-DO: also fix this to work with any number of cell types, add if statement, if only two groups get difference otherwise get SD
    # Figure out which clusters with positive labels have junctions in them with diff < 0.1 
    relabel_clusts = cell_type_psi_df[(cell_type_psi_df["sample_label"] == "positive") & (cell_type_psi_df["difference"] < 0.1)].Cluster.unique()
    # make new column named "true_label" which is the same as sample_label but for clusters that in relabel_clusts rename them to negative 
    cell_type_psi_df["true_label"] = cell_type_psi_df["sample_label"]
    cell_type_psi_df.loc[cell_type_psi_df["Cluster"].isin(relabel_clusts), "true_label"] = "negative"
    cell_type_psi_df.sort_values(by = ["new_junction_id_index"], inplace = True)
    print(cell_type_psi_df.true_label.value_counts(), cell_type_psi_df.sample_label.value_counts())

    # make dataframe using the following columns 
    sim_junc_counts_flat = pd.DataFrame({"cell_id_index": sim_juncs_counts.row, "new_junction_id_index": sim_juncs_counts.col, "new_junc_count": sim_juncs_counts.data})
    # also add new cell type column 
    sim_junc_counts_flat["new_cell_type"] = np.array(cell_types[sim_junc_counts_flat["cell_id_index"]])

    # Update junction counts in final_data object to be the simulated counts 
    final_data = final_data.merge(sim_junc_counts_flat, on = ["cell_id_index", "new_junction_id_index"])
    final_data.head()

    sim_data = final_data.copy() 
    # drop the old junction counts and junction id index
    sim_data.drop(columns = ["junc_count", "junction_id_index"], inplace = True)
    # rename columns new_junction_id_index and new_junc_count to junction_id_index and junc_count
    sim_data.rename(columns = {"new_junction_id_index": "junction_id_index", "new_junc_count": "junc_count"}, inplace = True)
    
    # update cluster_count to be the sum of junction_id_index in each Cluster for each cell
    new_clust_counts = sim_data.groupby(["cell_id_index", "Cluster"]).agg({"junc_count": "sum"}).reset_index()
    # update column to be cluster_count 
    new_clust_counts.rename(columns = {"junc_count": "Cluster_Counts"}, inplace = True)
    sim_data.drop(columns = ["Cluster_Counts"], inplace = True)
    # merge new_clust_counts with sim_data
    sim_data = sim_data.merge(new_clust_counts, on = ["cell_id_index", "Cluster"])

    # update juncratio 
    sim_data["clustminjunc"] = sim_data["Cluster_Counts"] - sim_data["junc_count"]
    sim_data["junc_ratio"] = sim_data["junc_count"] / sim_data["Cluster_Counts"]

    # juncs clusters labels
    juncs_labels = cell_type_psi_df[["new_junction_id_index", "Cluster", "true_label"]]
    juncs_labels = juncs_labels.drop_duplicates()

    # rename new_junction_id_index to junction_id_index 
    juncs_labels.rename(columns = {"new_junction_id_index": "junction_id_index"}, inplace = True)

    # merge with sim_data
    sim_data = sim_data.merge(juncs_labels, on = ["junction_id_index", "Cluster"])

    # Prepare tensors for the model
    cell_index_tensor, junc_index_tensor, my_data = llc.make_torch_data(final_data, **float_type)

    clust_labels_only = cluster_labels[["Cluster", "true_label"]].drop_duplicates()
    simple_data = sim_data[["cell_id_index", "Cluster", "cell_type", "junction_id_index", "junc_ratio", "junc_count", "Cluster_Counts"]]
    # merge with clust_labels_only 
    simple_data = simple_data.merge(clust_labels_only, on = ["Cluster"])
    
    # update cell_type column in simple data to what it is in cell_ids_conversion using cell_id_index and make sure it's a category
    simple_data.drop(columns = ["cell_type"], inplace = True)
    simple_data = simple_data.merge(cell_ids_conversion, on = ["cell_id_index"])
    simple_data["cell_type"] = simple_data["cell_type"].astype('category')
    
    print(f"Use simple_data object for visualization")
    print(simple_data.head())
    print("Data successfully simulated and prepared!")

    return cell_index_tensor, junc_index_tensor, my_data, final_data, simple_data
