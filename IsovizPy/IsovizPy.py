import gffutils
import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm

# Load Gencode GTF and create a database
def create_db(gtf_file, db_name='gencode_mm10.db'):
    return gffutils.create_db(
        gtf_file, dbfn=db_name, force=True, keep_order=True, 
        merge_strategy='merge', sort_attribute_values=True, 
        disable_infer_genes=True, disable_infer_transcripts=True
    )

# Function to check annotation status for 5' and 3' sides of a junction
def check_junction_annotation(splice_junctions, db, tolerance=1):
    junction_labels = []
    for junction in tqdm(splice_junctions):
        chrom = junction['chrom']
        start = junction['start']
        end = junction['end']
        strand = junction.get('strand', '+')
        label_5_prime, label_3_prime = "unannotated on 5'", "unannotated on 3'"
        position_off_5_prime, position_off_3_prime = None, None

        transcripts = list(db.region(region=(chrom, start - 5000, end + 5000), featuretype='transcript'))

        for transcript in transcripts:
            exons = list(db.children(transcript, featuretype='exon', order_by='start'))

            if strand == '+':
                for exon in exons:
                    if abs(exon.end - start) <= tolerance:
                        label_5_prime = "annotated on 5'"
                        position_off_5_prime = exon.end - start
                        break
                for exon in exons:
                    if abs(exon.start - end) <= tolerance:
                        label_3_prime = "annotated on 3'"
                        position_off_3_prime = exon.start - end
                        break
            elif strand == '-':
                for exon in exons:
                    if abs(exon.start - end) <= tolerance:
                        label_5_prime = "annotated on 5'"
                        position_off_5_prime = exon.start - end
                        break
                for exon in exons:
                    if abs(exon.end - start) <= tolerance:
                        label_3_prime = "annotated on 3'"
                        position_off_3_prime = exon.end - start
                        break

        junction_labels.append({
            'junction': f"{chrom}:{start}-{end}",
            'strand': strand,
            'label_5_prime': label_5_prime,
            'label_3_prime': label_3_prime,
            'position_off_5_prime': position_off_5_prime,
            'position_off_3_prime': position_off_3_prime
        })
    return junction_labels

# Function to fetch transcript and exon/CDS/UTR information
def fetch_transcripts_and_annotations(splice_junctions, db):
    transcript_data = {}
    for junction in splice_junctions:
        chrom = junction['chrom']
        start = junction['start']
        end = junction['end']
        strand = junction.get('strand', '+')
        junction_name = junction['name']
        transcripts = db.region(region=(chrom, start - 5000, end + 5000), featuretype='transcript')

        transcript_data[junction_name] = []
        for transcript in transcripts:
            exons = list(db.children(transcript, featuretype='exon', order_by='start'))
            cds = list(db.children(transcript, featuretype='CDS', order_by='start'))
            transcript_data[junction_name].append({
                'transcript_id': transcript.id,
                'exons': [(exon.start, exon.end) for exon in exons],
                'cds': [(cd.start, cd.end) for cd in cds],
                'strand': transcript.strand
            })
    return transcript_data

# Function to determine genomic region for plotting
def determine_region_boundaries(splice_junctions):
    min_coord = min([junction['start'] for junction in splice_junctions])
    max_coord = max([junction['end'] for junction in splice_junctions])
    return min_coord - 1000, max_coord + 1000

# Convert junction_id to structured format
def convert_junction_ids(df):
    splice_junctions = []
    for idx, row in df.iterrows():
        chrom, start, end, strand = row['junction_id'].split('_')
        splice_junctions.append({'chrom': chrom, 'start': int(start), 'end': int(end), 'name': f"junction_{idx + 1}", 'strand': strand})
    return splice_junctions

# Function to plot exons, CDS, and splice junctions with dynamic figure sizing
def plot_exons_and_junctions(db, transcript_data, splice_junctions, region_start, region_end, base_width=8, trans_height=0.5, show_junc_lines=True, junc_color="grey", filename="plot.pdf"):

    # Calculate height based on the number of transcripts and junctions
    num_transcripts = sum(len(transcripts) for transcripts in transcript_data.values())
    num_junctions = len(splice_junctions)
    base_height = 1  # Minimum height to start with
    height_per_transcript = trans_height
    height_per_junction = 0.05

    # Calculate dynamic height with some buffer space
    dynamic_height = base_height + (num_transcripts * height_per_transcript) + (num_junctions * height_per_junction)
    
    # Set the figure size
    fig, ax = plt.subplots(figsize=(base_width, dynamic_height))

    y_offset = 0
    plotted_transcripts, labels_added = set(), {'Exon': False, 'CDS': False, 'Intron': False, 'Junction': False}
    
    # Retrieve gene information from the first transcript
    first_transcript_id = next(iter(transcript_data.values()))[0]['transcript_id']
    gene = next(db.parents(first_transcript_id, featuretype='gene'))
    gene_id = gene.id
    gene_name = gene.attributes.get('gene_name', [gene_id])[0]  # Use gene ID if no name is found
    gene_strand = gene.strand

    # Plot each transcript's exons, CDS, and introns
    for junction_name, transcripts in transcript_data.items():
        for transcript in transcripts:
            transcript_id = transcript['transcript_id']
            if transcript_id in plotted_transcripts:
                continue
            plotted_transcripts.add(transcript_id)

            # Plot exons and CDS
            for exon_start, exon_end in transcript['exons']:
                ax.add_patch(plt.Rectangle((exon_start, y_offset - 0.3), exon_end - exon_start, 0.6, color='blue', 
                                           label='Exon' if not labels_added['Exon'] else ""))
                labels_added['Exon'] = True

            for cds_start, cds_end in transcript['cds']:
                ax.add_patch(plt.Rectangle((cds_start, y_offset - 0.3), cds_end - cds_start, 0.6, color='green', 
                                           label='CDS' if not labels_added['CDS'] else ""))
                labels_added['CDS'] = True

            # Connect exons with a dashed intron line
            for i in range(len(transcript['exons']) - 1):
                exon_end = transcript['exons'][i][1]
                next_exon_start = transcript['exons'][i + 1][0]
                ax.plot([exon_end, next_exon_start], [y_offset, y_offset], color='black', lw=1, ls='--', 
                        label='Intron' if not labels_added['Intron'] else "")
                labels_added['Intron'] = True

            # Annotate the transcript ID
            ax.text(region_end + 20, y_offset, transcript_id, verticalalignment='center', fontsize=10)

            y_offset += 1  # Increment offset for the next transcript

    # Leave space between transcripts and junctions
    y_offset += 1

    # Plot junctions below the transcripts
    for junction in splice_junctions:
        junction_start, junction_end = (junction['end'], junction['start']) if junction.get('strand', '+') == '-' else (junction['start'], junction['end'])

        # Plot the grey line for each junction
        ax.plot([junction_start, junction_end], [y_offset, y_offset], color=junc_color, lw=2, label='Junction' if not labels_added['Junction'] else "")
        labels_added['Junction'] = True

        if show_junc_lines:
            # Plot vertical lines at the junction start and end points
            ax.axvline(x=junction_start, color='red', linestyle='--', lw=1)
            ax.axvline(x=junction_end, color='purple', linestyle='--', lw=1)

        # Display the full junction coordinates as text
        ax.text((junction_start + junction_end) / 2, y_offset - 0.9, f"{junction['chrom']}:{junction_start}-{junction_end}", 
                ha='center', color='black', fontsize=7)

        y_offset += 1.5  # Leave extra space for the next junction

    # Customize plot appearance
    ax.set_xlim([region_start, region_end])
    ax.set_ylim([-1, y_offset])
    ax.set_xlabel("Genomic Position", fontsize=12)
    ax.set_title(f"Splice Junctions and Exons for Gene {gene_name} (ID: {gene_id}, Strand: {gene_strand})", fontsize=12)
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=7, title_fontsize=7)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()
