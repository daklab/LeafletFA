import gffutils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from datetime import datetime
from pyfaidx import Fasta

# Load Gencode GTF and create a database
def create_db(gtf_file, db_name="gencode_mm10.db"):
    return gffutils.create_db(
        gtf_file,
        dbfn=db_name,
        force=True,
        keep_order=True,
        merge_strategy="merge",
        sort_attribute_values=True,
        disable_infer_genes=True,
        disable_infer_transcripts=True,
    )

# Function to determine genomic region for plotting
def determine_region_boundaries(splice_junctions):
    min_coord = min([junction["start"] for junction in splice_junctions])
    max_coord = max([junction["end"] for junction in splice_junctions])
    return min_coord - 1000, max_coord + 1000


# Convert junction_id to structured format
def convert_junction_ids(df):
    splice_junctions = []
    for idx, row in df.iterrows():
        chrom, start, end, strand = row["junction_id"].split("_")
        splice_junctions.append(
            {
                "chrom": chrom,
                "start": int(start),
                "end": int(end),
                "name": f"junction_{idx + 1}",
                "strand": strand,
            }
        )
    return splice_junctions


def convert_junction_ids(df):
    splice_junctions = []
    for idx, row in df.iterrows():
        chrom, start, end, strand = row["junction_id"].split("_")
        splice_junctions.append(
            {
                "chrom": chrom,
                "start": int(start),
                "end": int(end),
                "name": f"junction_{idx + 1}",
                "strand": strand,
                "usage_ratio": row["usage_ratio"],  # Add usage_ratio to each junction dictionary
            }
        )
    return splice_junctions


def plot_exons_and_junctions(
    db,
    atse, 
    transcript_data,
    splice_junctions,
    region_start,
    region_end,
    base_width=8,
    trans_height=0.5,
    junc_color="grey",
    show_usage=False,
    show_junc_lines=True,
    colorbar_pad=0.5,
    filename=None,
):
    """
    Plots exons, CDS, and splice junctions for a given gene, with annotations for transcript names and types.
    """
    # Sort transcript_data by transcript type
    # Define custom order for sorting (e.g., protein-coding first, then NMD, etc.)
    type_order = {"protein_coding": 1, "nonsense_mediated_decay": 2, "other": 3}

    sorted_transcripts = sorted(
        transcript_data.items(),
        key=lambda x: type_order.get(x[1]["transcript_type"], type_order["other"]),
    )

    # Calculate height based on the number of transcripts and junctions
    num_transcripts = len(transcript_data)
    num_junctions = len(splice_junctions)
    base_height = 1  # Minimum height to start with
    height_per_transcript = trans_height
    height_per_junction = 0.05

    # Calculate dynamic height with some buffer space
    dynamic_height = (
        base_height
        + (num_transcripts * height_per_transcript)
        + (num_junctions * height_per_junction)
    )

    # Set the figure size
    fig, ax = plt.subplots(figsize=(base_width, dynamic_height))

    y_offset = 0
    plotted_transcripts, labels_added = set(), {
        "Exon": False,
        "CDS": False,
        "Intron": False,
        "Junction": False,
    }

    # Retrieve gene information from the first transcript
    first_transcript_id = next(iter(transcript_data.values()))["transcript_id"]
    gene = next(db.parents(first_transcript_id, featuretype="gene"))

    gene_id = gene.id
    gene_name = gene.attributes.get("gene_name", [gene_id])[
        0
    ]  # Use gene ID if no name is found
    gene_strand = gene.strand

    # Plot each transcript's exons, CDS, and introns
    for transcript_id, transcript in sorted_transcripts:

        if transcript_id in plotted_transcripts:
            continue
        plotted_transcripts.add(transcript_id)

        # Plot exons and CDS
        for exon_start, exon_end in transcript["exons"]:
            ax.add_patch(
                plt.Rectangle(
                    (exon_start, y_offset - 0.3),
                    exon_end - exon_start,
                    0.6,
                    color="blue",
                    label="Exon" if not labels_added["Exon"] else "",
                )
            )
            labels_added["Exon"] = True

        for cds_start, cds_end in transcript["cds"]:
            ax.add_patch(
                plt.Rectangle(
                    (cds_start, y_offset - 0.3),
                    cds_end - cds_start,
                    0.6,
                    color="green",
                    label="CDS" if not labels_added["CDS"] else "",
                )
            )
            labels_added["CDS"] = True

        # Connect exons with a dashed intron line
        for i in range(len(transcript["exons"]) - 1):
            exon_end = transcript["exons"][i][1]
            next_exon_start = transcript["exons"][i + 1][0]
            ax.plot(
                [exon_end, next_exon_start],
                [y_offset, y_offset],
                color="black",
                lw=1,
                ls="--",
                label="Intron" if not labels_added["Intron"] else "",
            )
            labels_added["Intron"] = True

        # Annotate with transcript name and type
        transcript_label = (
            f"{transcript['transcript_name']} ({transcript['transcript_type']})"
        )
        ax.text(
            region_end + 20,
            y_offset,
            transcript_label,
            verticalalignment="center",
            fontsize=10,
        )

        y_offset += 1  # Move up for the next transcript

    # Leave space between transcripts and junctions
    y_offset += 1

    # Normalize usage ratios and create a color map
    if splice_junctions:
        usage_ratios = [junction["usage_ratio"] for junction in splice_junctions]
        norm = mcolors.Normalize(vmin=min(usage_ratios), vmax=max(usage_ratios))
        cmap = cm.copper  # Use Reds colormap
    else:
        norm = mcolors.Normalize(
            vmin=0, vmax=1
        )  # Placeholder norm if no junctions exist

    # Plot junctions below the transcripts
    for junction in splice_junctions:
        junction_start, junction_end = (junction["start"], junction["end"])

        # Get color based on usage_ratio
        color = cmap(norm(junction["usage_ratio"])) if show_usage else junc_color

        # Plot the line for each junction with usage_ratio-based color if show_usage is True
        ax.plot(
            [junction_start, junction_end],
            [y_offset, y_offset],
            color=color,
            lw=2,
            label="Junction" if not labels_added["Junction"] else "",
        )
        labels_added["Junction"] = True

        if show_junc_lines:
            # Plot vertical lines at the junction start and end points
            ax.axvline(x=junction_start, color="red", linestyle="--", lw=1)
            ax.axvline(x=junction_end, color="purple", linestyle="--", lw=1)

        # Display the full junction coordinates as text
        ax.text(
            (junction_start + junction_end) / 2,
            y_offset - 0.9,
            f"{junction['chrom']}:{junction_start}-{junction_end}",
            ha="center",
            color="black",
            fontsize=7,
        )

        y_offset += 1.5  # Leave extra space for the next junction

    # Customize plot appearance
    ax.set_xlim([region_start, region_end])
    ax.set_ylim([-1, y_offset])
    ax.set_xlabel("Genomic Position", fontsize=12)
    ax.set_title(
        f"Splice Junctions and Exons for Gene {gene_name} \n(ID: {gene_id}, Strand: {gene_strand} \nATSE: {atse})",
        fontsize=12,
    )
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(loc="upper left", fontsize=6, title_fontsize=6)

    if show_usage:
        # Add colorbar for usage_ratio
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Array is only needed for colorbar; empty here
        plt.colorbar(
            sm, ax=ax, orientation="horizontal", pad=colorbar_pad, label="% Usage"
        )

    # Automatically generate filename if not provided
    if filename is None:
        date_str = datetime.now().strftime("%Y%m%d")
        usage_str = "usage" if show_usage else "no_usage"
        lines_str = "lines" if show_junc_lines else "no_lines"
        filename = f"{gene_name}_{usage_str}_{lines_str}_{date_str}.pdf"

    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    print(f"Plot saved to {filename}!")
    plt.show()


def fetch_transcripts_and_annotations(db, transcript_ids):
    """
    Retrieves transcript data and organizes it with exons, CDS, and strand information,
    along with transcript name and type.
    """
    transcript_data = {}

    for transcript_id in transcript_ids:
        try:
            # Fetch transcript from the database
            transcript = db[transcript_id]
            # Retrieve exons and CDS features
            exons = list(db.children(transcript, featuretype="exon", order_by="start"))
            cds = list(db.children(transcript, featuretype="CDS", order_by="start"))

            # Extract transcript name and type (if available)
            transcript_name = transcript.attributes.get(
                "transcript_name", [transcript_id]
            )[0]
            transcript_type = transcript.attributes.get("transcript_type", ["Unknown"])[
                0
            ]
            chromosome = transcript.chrom

            # Store transcript data with additional information
            transcript_data[transcript_id] = {
                "transcript_id": transcript_id,
                "transcript_name": transcript_name,
                "transcript_type": transcript_type,
                "chromosome": chromosome,
                "exons": [(exon.start, exon.end) for exon in exons],
                "cds": [(cd.start, cd.end) for cd in cds],
                "strand": transcript.strand,
            }
        except KeyError:
            print(f"Transcript '{transcript_id}' not found in the database.")

    return transcript_data


def fetch_transcripts_for_gene(db, gene_name):
    """
    Fetches all transcripts associated with a given gene name.
    """
    gene = None
    for feature in db.features_of_type("gene"):
        if feature.attributes.get("gene_name", [None])[0] == gene_name:
            gene = feature
            break

    if gene:
        # Retrieve all transcripts associated with this gene
        gene_transcripts = [
            transcript.id for transcript in db.children(gene, featuretype="transcript")
        ]
        return gene_transcripts
    else:
        print(f"Gene '{gene_name}' not found in the database.")
        return []


def get_transcripts_for_transcripts(db, transcript_list):
    """
    Retrieves the transcript features from the database for a given list of transcript IDs,
    maintaining the order of the provided list.

    Parameters:
        db (gffutils.FeatureDB): The gffutils database.
        transcript_list (list): A list of transcript IDs to retrieve.

    Returns:
        list: A list of transcript features in the same order as transcript_list.
    """
    # Initialize an empty list to hold the transcript features
    transcripts = []

    # Iterate over each transcript ID in the provided list
    for transcript_id in transcript_list:
        try:
            # Retrieve the transcript feature from the database
            transcript = db[transcript_id]
            # Add the transcript to the list, preserving the original order
            transcripts.append(transcript)
        except KeyError:
            print(f"Transcript '{transcript_id}' not found in the database.")

    return transcripts


def determine_region_boundaries_from_transcripts(transcript_data):
    """
    Determines the region boundaries for plotting based on the transcript data.
    """
    region_start = min(
        exon[0]
        for transcript in transcript_data.values()
        for exon in transcript["exons"]
    )
    region_end = max(
        exon[1]
        for transcript in transcript_data.values()
        for exon in transcript["exons"]
    )
    return region_start, region_end


def create_transform_function(intervals_to_compress, compression_factor):
    intervals = sorted(intervals_to_compress)
    cumulative_shifts = []
    total_shift = 0
    for start, end in intervals:
        original_length = end - start
        compressed_length = original_length * compression_factor
        shift = original_length - compressed_length
        total_shift += shift
        cumulative_shifts.append((start, end, total_shift))

    def transform_x(x):
        shift = 0
        for start, end, cumulative_shift in cumulative_shifts:
            if x >= end:
                shift = cumulative_shift
            elif x >= start:
                shift = cumulative_shift - (end - x) * (1 - compression_factor)
                break
            else:
                break
        return x - shift

    def reverse_transform_x(x):
        shift = 0
        for start, end, cumulative_shift in cumulative_shifts:
            if x >= end:
                shift = cumulative_shift
            elif x >= start:
                shift = cumulative_shift - (end - x) * (1 - compression_factor)
                break
            else:
                break
        return x + shift

    return transform_x, reverse_transform_x

def plot_isoforms(
    db,
    transcript_data,
    region_start,
    region_end,
    transcript_order=None, #list of transcript_names
    base_width=8,
    fixed_height=False, 
    fixed_height_value=8,
    trans_height=0.5,
    filename=None,
    introns_list=None,
    show_transcript_type=False,
    plot_shared_intron_lines=False,
    plot_introns_as_blocks=False,
    compression_factor=1,
    num_ticks=5,
    trans_label=50
):
    """
    Plots isoforms with exons, CDS, and introns for each transcript in transcript_data,
    labeling transcripts with their name and type.
    """
    # Calculate dynamic height based on the number of transcripts
    num_transcripts = len(transcript_data)
    dynamic_height = 2 + (num_transcripts * trans_height)
    
    if fixed_height:
        dynamic_height = fixed_height_value

    print(f"(dynamic_height", dynamic_height)
    fig, ax = plt.subplots(figsize=(base_width, dynamic_height))
    
    y_offset = 0
    labels_added = {"Exon": False, "CDS": False, "Intron": False}
    # Retrieve gene information from the first transcript
    first_transcript_id = next(iter(transcript_data.values()))["transcript_id"]
    gene = next(db.parents(first_transcript_id, featuretype="gene"))

    gene_id = gene.id
    gene_name = gene.attributes.get("gene_name", [gene_id])[
        0
    ]  # Use gene ID if no name is found
    gene_strand = gene.strand
    xlab = "Genomic Coordinates"

    if compression_factor != 1:
        assert introns_list is not None, "Introns list must be provided for compression"
        transform_x, reverse_transform_x = create_transform_function(
            introns_list, compression_factor
        )
        xlab = "Genomic Coordinates with Compressed Shared Introns"

    else:
        transform_x = lambda x: x
        reverse_transform_x = lambda x: x

    # Plot each transcript's exons, CDS, and introns
    for transcript_id in transcript_order:
        transcript_info = transcript_data[transcript_id]

        # Plot exons
        for exon_start, exon_end in transcript_info["exons"]:
            exon_start = transform_x(exon_start)
            exon_end = transform_x(exon_end)
            ax.add_patch(
                plt.Rectangle(
                    (exon_start, y_offset - 0.3),
                    exon_end - exon_start,
                    0.6,
                    color="blue",
                    label="Exon" if not labels_added["Exon"] else "",
                )
            )
            labels_added["Exon"] = True

        # Plot CDS
        for cds_start, cds_end in transcript_info["cds"]:
            cds_start = transform_x(cds_start)
            cds_end = transform_x(cds_end)
            ax.add_patch(
                plt.Rectangle(
                    (cds_start, y_offset - 0.3),
                    cds_end - cds_start,
                    0.6,
                    color="green",
                    label="CDS" if not labels_added["CDS"] else "",
                )
            )
            labels_added["CDS"] = True

        # Connect exons with dashed intron lines
        for i in range(len(transcript_info["exons"]) - 1):
            exon_end = transcript_info["exons"][i][1]
            next_exon_start = transcript_info["exons"][i + 1][0]
            exon_end = transform_x(exon_end)
            next_exon_start = transform_x(next_exon_start)

            ax.plot(
                [exon_end, next_exon_start],
                [y_offset, y_offset],
                color="black",
                lw=1,
                ls="--",
                label="Intron" if not labels_added["Intron"] else "",
            )
            labels_added["Intron"] = True

        # Annotate with transcript name and type
        if show_transcript_type:
            transcript_label = f"{transcript_info['transcript_name']} ({transcript_info['transcript_type']})"

        else:
            transcript_label = f"{transcript_info['transcript_name']}"

        ax.text(
            transform_x(region_end) + trans_label,
            y_offset,
            transcript_label,
            verticalalignment="center",
            fontsize=10,
        )

        y_offset += 1  # Move up for the next transcript

    y_offset += 1  # Move up for the next transcript
    
    if introns_list is not None:
        if plot_shared_intron_lines and not plot_introns_as_blocks:
            # plot introns as vertical lines
            for intron_coords in introns_list:
                # Plot vertical lines at the junction start and end points
                intron_start = transform_x(intron_coords[0])
                intron_end = transform_x(intron_coords[1])
                ax.axvline(x=intron_start, color="red", linestyle="--", lw=1)
                ax.axvline(x=intron_end, color="purple", linestyle="--", lw=1)

        elif plot_introns_as_blocks and not plot_shared_intron_lines:
            # Plot introns as grey blocks
            for intron_coords in introns_list:
                intron_start = transform_x(intron_coords[0])
                intron_end = transform_x(intron_coords[1])
                ax.add_patch(
                    plt.Rectangle(
                    (intron_start, y_offset - 0.5),  # Position below last transcript row
                    intron_end - intron_start,
                    0.4,
                    color="grey",
                    alpha=0.5,
                    label="Intron Block" if not labels_added.get("Intron Block") else "",
                )
                )
                labels_added["Intron Block"] = True

    # Customize plot appearance
    ax.set_xlim([transform_x(region_start), transform_x(region_end)])
    ax.set_ylim([-1, y_offset])
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_title(
        f"Isoforms (Exons and CDS) for Gene {gene_name} \n(ID: {gene_id}, Strand: {gene_strand})",
        fontsize=12,
    )
    original_coordinate_ticks = np.linspace(
        round(region_start, -2), round(region_end, -2), num=num_ticks
    )
    original_coordinate_ticks = [int(x) for x in original_coordinate_ticks]
    new_coord_ticks = [transform_x(x) for x in original_coordinate_ticks]

    ax.set_xticks(new_coord_ticks)
    ax.set_xticklabels(original_coordinate_ticks)

    ax.set_yticks([])
    
    # ax.legend(loc="upper left", fontsize=6, title_fontsize=6)

    ax.legend(
    loc="upper left",  # Position the legend relative to the bbox
    fontsize=6, 
    title_fontsize=6,
    bbox_to_anchor=(-0.15, 1), # Adjusts position outside the axes
    borderaxespad=0  # Adjust padding
    )

    # Save plot
    if filename is None:
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{gene_name}_isoforms_plot_{date_str}_compression_{str(compression_factor)}.pdf"

    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    print(f"Plot saved to {filename}!")


def get_intervals_not_overlapped_by_exons(region_start, region_end, exons):
    """
    Given a genomic region and a list of exon intervals, return the intervals within the region
    that are not overlapped by any exon intervals.

    Parameters:
    - region_start (int): The start position of the region.
    - region_end (int): The end position of the region.
    - exons (list of tuples): A list of (start, end) tuples representing exon intervals.

    Returns:
    - list of tuples: Intervals not overlapped by exons within the specified region.
    """
    # Filter exons that overlap with the region and adjust their boundaries
    overlapping_exons = []
    for exon_start, exon_end in exons:
        if exon_end < region_start or exon_start > region_end:
            continue  # Exon does not overlap with the region
        # Adjust exon boundaries to be within the region
        start = max(exon_start, region_start)
        end = min(exon_end, region_end)
        overlapping_exons.append((start, end))

    # Sort exons by their start positions
    overlapping_exons.sort()

    result = []
    current_pos = region_start

    for exon_start, exon_end in overlapping_exons:
        if current_pos < exon_start:
            # There's a gap between the current position and the start of the exon
            result.append((current_pos, exon_start - 1))
        # Move current position to after the current exon
        current_pos = max(current_pos, exon_end + 1)

    if current_pos <= region_end:
        # Add the remaining region after the last exon
        result.append((current_pos, region_end))

    return result
