def visualize_local_events(dat, junc_id=None, cluster_id=None, p_usage_ratio=True):

    # Filter dat based on junc_id, or cluster_id 
    if junc_id is not None:
        dat = dat[dat.Cluster == dat[dat.junction_id == junc_id].Cluster.values[0]]
    elif cluster_id is not None:
        dat = dat[dat.Cluster == cluster_id]

    # Get junctions
    juncs = dat[["chrom", "chromStart", "chromEnd", "strand", "intron_length", "counts_total", "Start_b", "End_b", "exon_id"]]
    juncs = juncs.drop_duplicates()
    juncs["junc_usage_ratio"] = juncs["counts_total"] / juncs["counts_total"].sum()

    # Sort junctions based on strand
    if juncs.strand.values[0] == "+":
        juncs = juncs.sort_values("chromStart")
    else:
        juncs = juncs.sort_values("chromEnd", ascending=False)

    # Convert genomic coordinates to kilobases
    juncs["chromStart_kb"] = basepair_to_kilobase(juncs["chromStart"])
    juncs["chromEnd_kb"] = basepair_to_kilobase(juncs["chromEnd"])
    # Convert exon coordinates to kilobases
    juncs["Start_b"] = basepair_to_kilobase(juncs["Start_b"])
    juncs["End_b"] = basepair_to_kilobase(juncs["End_b"])

    print(juncs[["chrom", "chromStart_kb", "chromEnd_kb", "strand", "intron_length", "counts_total", "Start_b", "End_b", "exon_id"]])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, len(juncs) * 0.5))

    # Unique exons
    exon_ids = juncs.exon_id.unique()
    colors = plt.cm.tab20.colors
    color_dict = {exon_id: colors[i % len(colors)] for i, exon_id in enumerate(exon_ids)}
    cmap = plt.get_cmap('plasma')

    # Plot junctions as lines with varying colors based on usage ratio
    for i, (_, junc) in enumerate(juncs.iterrows()):
        color = cmap(junc["junc_usage_ratio"])
        ax.plot([junc["chromStart_kb"], junc["chromEnd_kb"]], [i, i], color=color)

    # Add vertical lines at unique start_b and end_b positions for each exon with dashed line
    for exon_id, group in juncs.groupby("exon_id"):
        for _, exon in group.iterrows():
            ax.axvline(x=exon["Start_b"], color=color_dict[exon_id], linestyle="--")
            ax.axvline(x=exon["End_b"], color=color_dict[exon_id], linestyle="--")

    # Set labels and title 
    ax.set_xlabel(f"Genomic Position on chr{juncs.chrom.values[0]} ({juncs.strand.values[0]}) [Kilobases]")
    ax.set_yticks([])  # Remove y-axis ticks
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))  # Disable scientific notation

    # Plot junction usage ratios if requested
    if p_usage_ratio:
        for i, (_, junc) in enumerate(juncs.iterrows()):
            ax.text(junc["chromEnd_kb"], i, f'{junc["junc_usage_ratio"]:.3f}', verticalalignment='center', fontsize=8)

    # Check if gene_name column is in dat and use it as title if it is otherwise use gene_id
    if "gene_name" in dat.columns:
        ax.set_title(f"Visualization of Junctions in Cluster {dat.Cluster.values[0]} in the Gene {dat.gene_name.values[0]}")
    else:
        ax.set_title(f"Visualization of Junctions in Cluster {dat.Cluster.values[0]} in the Gene {dat.gene_id.values[0]}")

    print("The junction of interest is " + str(junc_id))
    plt.show()
