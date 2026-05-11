import pandas as pd  
import gzip 

def read_gtf_file(file_path, feature_type='all'):
    """
    Read a GTF file (supports gzip-compressed .gtf.gz files) and filter based on the specified feature type.

    Parameters:
    - file_path (str): Path to the GTF file (.gtf or .gtf.gz).
    - feature_type (str): Feature type to filter ('all', 'exon', 'transcript', 'gene_id').

    Returns:
    - pandas.DataFrame: DataFrame representing GTF entries matching the filter criteria.
    """
    data = {
        'seqname': [],
        'source': [],
        'feature': [],
        'start': [],
        'end': [],
        'score': [],
        'strand': [],
        'frame': [],
        'attributes': []
    }
    
    open_func = gzip.open if file_path.endswith('.gz') else open
    
    with open_func(file_path, 'rt') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comment lines
            fields = line.strip().split('\t')
            attributes = dict(item.strip().split(' ') for item in fields[8].strip().split(';') if item.strip())
            
            if (feature_type == 'all' or
                (feature_type == 'exon' and fields[2] == 'exon') or
                (feature_type == 'transcript' and fields[2] == 'transcript') or
                (feature_type == 'gene_id' and 'gene_id' in attributes)):
                
                data['seqname'].append(fields[0])
                data['source'].append(fields[1])
                data['feature'].append(fields[2])
                data['start'].append(int(fields[3]))
                data['end'].append(int(fields[4]))
                data['score'].append(fields[5])
                data['strand'].append(fields[6])
                data['frame'].append(fields[7])
                data['attributes'].append(attributes)
    
    return pd.DataFrame(data)
