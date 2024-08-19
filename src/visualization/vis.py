import matplotlib.pyplot as plt
import seaborn as sns

def get_clust_plot(clust, simple_data, num_cols=3, plot_states_arb=True):
    
    simple_data_junc = simple_data[simple_data["Cluster"] == clust]
    
    # make violin plot with jitter 
    print(simple_data_junc.cell_type.value_counts())
    sample_label = simple_data_junc.true_label.unique()[0]
    
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