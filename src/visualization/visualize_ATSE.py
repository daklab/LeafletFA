import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Helper function to convert base pairs to kilobases
def basepair_to_kilobase(bp):
    return bp / 1000

# Helper function to convert base pairs to a custom scale (10,000 by default)
def basepair_to_scaled(bp, scale=10000):
    return bp / scale

# Function to visualize local events in splicing data
def visualize_local_events(dat, junc_id=None, cluster_id=None, p_usage_ratio=True):
    
    # Filter dat based on junc_id, or cluster_id 
    if junc_id is not None:
        dat = dat[dat.Cluster == dat[dat.junction_id == junc_id].Cluster.values[0]]
    elif cluster_id is not None:
        dat = dat[dat.Cluster == cluster_id]

    # Get junctions and relevant columns
    juncs = dat[["chrom", "junc_start", "junc_end", "strand", "total_read_counts", "exon_start", "exon_end", "exon_id"]]
    juncs = juncs.drop_duplicates()
    juncs["junc_usage_ratio"] = juncs["total_read_counts"] / juncs["total_read_counts"].sum()

    # Sort junctions based on strand
    if juncs.strand.values[0] == "+":
        juncs = juncs.sort_values("junc_start")
    else:
        juncs = juncs.sort_values("junc_end", ascending=False)

    # Convert genomic coordinates to kilobases
    juncs["junc_start_kb"] = basepair_to_kilobase(juncs["junc_start"])
    juncs["junc_end_kb"] = basepair_to_kilobase(juncs["junc_end"])
    juncs["exon_start_kb"] = basepair_to_kilobase(juncs["exon_start"])
    juncs["exon_end_kb"] = basepair_to_kilobase(juncs["exon_end"])

    # Display the junctions
    print(juncs[["chrom", "junc_start_kb", "junc_end_kb", "strand", "total_read_counts", "exon_start_kb", "exon_end_kb", "exon_id"]])

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
        ax.plot([junc["junc_start_kb"], junc["junc_end_kb"]], [i, i], color=color)

    # Add vertical lines at unique exon_start and exon_end positions for each exon with dashed lines
    for exon_id, group in juncs.groupby("exon_id"):
        for _, exon in group.iterrows():
            ax.axvline(x=exon["exon_start_kb"], color=color_dict[exon_id], linestyle="--")
            ax.axvline(x=exon["exon_end_kb"], color=color_dict[exon_id], linestyle="--")

    # Set labels and title
    ax.set_xlabel(f"Genomic Position on chr{juncs.chrom.values[0]} ({juncs.strand.values[0]}) [Kilobases]")
    ax.set_yticks([])  # Remove y-axis ticks
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))  # Disable scientific notation

    # Plot junction usage ratios if requested
    if p_usage_ratio:
        for i, (_, junc) in enumerate(juncs.iterrows()):
            ax.text(junc["junc_end_kb"], i, f'{junc["junc_usage_ratio"]:.3f}', verticalalignment='center', fontsize=8)

    # Check if gene_name column is in dat and use it as title if available, otherwise use gene_id
    if "gene_name" in dat.columns:
        ax.set_title(f"Visualization of Junctions in Cluster {dat.Cluster.values[0]} in the Gene {dat.gene_name.values[0]}")
    else:
        ax.set_title(f"Visualization of Junctions in Cluster {dat.Cluster.values[0]} in the Gene {dat.gene_id.values[0]}")

    print("The junction of interest is " + str(junc_id))
    plt.show()

# Function to visualize splice graph with vertically aligned exons and rounded PSI values
def visualize_splice_graph(dat, cluster_id=None, scale_factor=10000, padding_factor=0.05):
    
    # Filter dat based on cluster_id 
    if cluster_id is not None:
        dat = dat[dat.Cluster == cluster_id]

    # Get junctions and relevant columns
    juncs = dat[["chrom", "junc_start", "junc_end", "strand", "total_read_counts", "exon_start", "exon_end", "exon_id"]]
    juncs = juncs.drop_duplicates()
    
    # Calculate junction usage ratio (PSI)
    juncs["junc_usage_ratio"] = juncs["total_read_counts"] / juncs["total_read_counts"].sum()

    # Sort junctions based on strand
    if juncs.strand.values[0] == "+":
        juncs = juncs.sort_values("junc_start")
    else:
        juncs = juncs.sort_values("junc_end", ascending=False)

    # Convert genomic coordinates to custom scale (e.g., 10,000 bp)
    juncs["junc_start_scaled"] = basepair_to_scaled(juncs["junc_start"], scale=scale_factor)
    juncs["junc_end_scaled"] = basepair_to_scaled(juncs["junc_end"], scale=scale_factor)
    juncs["exon_start_scaled"] = basepair_to_scaled(juncs["exon_start"], scale=scale_factor)
    juncs["exon_end_scaled"] = basepair_to_scaled(juncs["exon_end"], scale=scale_factor)

    # Determine the range of genomic coordinates for padding
    first_exon_start = juncs['exon_start_scaled'].min()
    last_exon_end = juncs['exon_end_scaled'].max()
    padding = (last_exon_end - first_exon_start) * padding_factor

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Unique exons
    exon_ids = juncs.exon_id.unique()
    colors = plt.cm.tab20.colors
    color_dict = {exon_id: colors[i % len(colors)] for i, exon_id in enumerate(exon_ids)}
    cmap = plt.get_cmap('plasma')

    # Set all exons to the same vertical level
    exon_y = 0

    # Plot exons as filled rectangles and junctions as dotted curvy lines
    for i, (_, junc) in enumerate(juncs.iterrows()):
        # Color for the exon
        exon_color = color_dict[junc["exon_id"]]

        # Plot filled rectangles for exons (vertically aligned at `exon_y`)
        ax.fill_between([junc["exon_start_scaled"], junc["exon_end_scaled"]], exon_y - 0.1, exon_y + 0.1, color=exon_color, alpha=0.6)

        # Get the junction usage ratio (PSI) value and map it to the colormap
        color = cmap(junc["junc_usage_ratio"])

        # Plot junctions as dotted curvy lines
        junction_curve_x = np.linspace(junc["junc_start_scaled"], junc["junc_end_scaled"], 100)
        junction_curve_y = exon_y + 0.3 * np.sin(np.linspace(0, np.pi, 100))  # Sinusoidal curve for curvy lines
        ax.plot(junction_curve_x, junction_curve_y, linestyle=":", color=color)

    # Add vertical lines at unique exon start and end positions with dashed lines
    for exon_id, group in juncs.groupby("exon_id"):
        for _, exon in group.iterrows():
            ax.axvline(x=exon["exon_start_scaled"], color=color_dict[exon_id], linestyle="--")
            ax.axvline(x=exon["exon_end_scaled"], color=color_dict[exon_id], linestyle="--")

    # Set labels and title
    ax.set_xlabel(f"Genomic Position on chr{juncs.chrom.values[0]} ({juncs.strand.values[0]}) [scaled by {scale_factor} bp]")
    ax.set_xlim(first_exon_start - padding - 100, last_exon_end + padding + 100)
    ax.set_yticks([])  # Remove y-axis ticks
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))  # Disable scientific notation

    # Plot junction usage ratios (PSI) on top of junctions, rounded to 3 digits
    for i, (_, junc) in enumerate(juncs.iterrows()):
        mid_point = (junc["junc_start_scaled"] + junc["junc_end_scaled"]) / 2
        ax.text(mid_point, exon_y + 0.35, f'{junc["junc_usage_ratio"]:.3f}', ha='center', fontsize=8)

    # Title based on gene_name or gene_id
    if "gene_name" in dat.columns:
        ax.set_title(f"Splice Graph for Cluster {dat.Cluster.values[0]} in Gene {dat.gene_name.values[0]} ({juncs.strand.values[0]})")
    else:
        ax.set_title(f"Splice Graph for Cluster {dat.Cluster.values[0]} in Gene {dat.gene_id.values[0]} ({juncs.strand.values[0]})")

    plt.show()

# Example usage:
# visualize_splice_graph(dat, cluster_id=3677, scale_factor=10000)