import pandas as pd
import logging
import multiprocessing as mp
import numpy as np
import os

# Initialize logging
logging.basicConfig(filename='pubmed_processing.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

file_path = '/app/dataset_with_abstracts.csv'
output_file = '/app/workspace/data/output/abstracts.csv'
CVD_DRUGS = ["Drug Therapy","Cardiovascular Agents", "Antihypertensive Agents", "Cardiovascular diseases",
    "Treatment of Cardiovascular Diseases",
    "Drugs for Primary and Secondary Cardiovascular Prevention",
    "Heart Failure",
    "Hypertension",
    "Myocardial Infarction",
    "Stroke Volume",
    "Statin Therapy",

    # ACE Inhibitors
    "ACE Inhibitors",
    "Captopril",
    "Enalapril",
    "Lisinopril",
    "Ramipril",

    
    # Angiotensin II Receptor Blockers (ARBs)
    "Angiotensin II Receptor Blockers",
    "Losartan",
    "Valsartan",
    "Irbesartan",
    "Candesartan",
    
    # Beta-Adrenergic Blockers
    "Beta-Adrenergic Blockers",
    "Atenolol",
    "Metoprolol",
    "Propranolol",
    "Carvedilol",
    
    # Calcium Channel Blockers
    "Calcium Channel Blockers",
    "Amlodipine",
    "Nifedipine",
    "Diltiazem",
    "Verapamil",
    
    # HMG-CoA Reductase Inhibitors (Statins)
    "HMG-CoA Reductase Inhibitors",
    'Hydroxymethylglutaryl-CoA Reductase Inhibitors',
    "Atorvastatin",
    "Simvastatin",
    "Rosuvastatin",
    "Pravastatin",
    
    # Platelet Aggregation Inhibitors (Antiplatelet Agents)
    "Platelet Aggregation Inhibitors",
    "Antiplatelet Agents",
    "Aspirin",
    "Clopidogrel",
    "Ticagrelor",
    "Dipyridamole",
    
    # Anticoagulants
    "Anticoagulants",
    "Warfarin",
    "Heparin",
    "Enoxaparin",
    "Apixaban",
    "Rivaroxaban",
    
    # Vasodilator Agents
    "Vasodilator Agents",
    "Nitroglycerin",
    "Hydralazine",
    "Isosorbide Dinitrate",
    
    # Diuretics
    "Diuretics",
    "Furosemide",
    "Hydrochlorothiazide",
    "Spironolactone",
    "Eplerenone",
    
    # Phosphodiesterase Inhibitors
    "Phosphodiesterase Inhibitors",
    "Milrinone",
    "Amrinone",
    "Inamrinone",
    
    # Direct Renin Inhibitors
    "Direct Renin Inhibitors",
    "Aliskiren",
    
    # If Channel Inhibitors
    "If Channel Inhibitors",
    "Ivabradine",
    
    # Neprilysin Inhibitors
    "Neprilysin Inhibitors",
    "Sacubitril/Valsartan",
    
    # Antiarrhythmic Agents
    "Antiarrhythmic Agents",
    "Amiodarone",
    "Sotalol",
    "Dronedarone",
    
    # Thrombolytic Agents
    "Thrombolytic Agents",
    "Tissue Plasminogen Activator",
    "Streptokinase",
    "Urokinase-Type Plasminogen Activator"
]

def process_chunk(df, chunk_idx):
    try:
        if 'MeshHeadings' not in df.columns or 'Language' not in df.columns:
            logging.error(f'Chunk {chunk_idx} is missing required columns.')
            return pd.DataFrame(columns=['PMID','ArticleTitle','MeshHeadings', 'AbstractText',''])

        empty_mesh_count = df['MeshHeadings'].isna().sum()
        df['MeshHeadings'] = df['MeshHeadings'].str.lower()
        language_counts = df['Language'].value_counts()
        #how many rows have number of references and references
        print(df.columns)

        df_filtered = df[~df['MeshHeadings'].isna() & (df['Language'] == 'eng')].copy()
        
        df_filtered['MeshHeadings'] = df_filtered['MeshHeadings'].apply(
            lambda x: x.split(';') if isinstance(x, str) else []
        )
                
        df_filtered = df_filtered[df_filtered.apply(lambda row: any(term in row['MeshHeadings'] or 
                                                                    term in row['ArticleTitle'] or 
                                                                    term in row['AbstractText']
                                                                    for term in CVD_DRUGS), axis=1)]

        df_filtered['MeshHeadings'] = df_filtered['MeshHeadings'].apply(lambda x: ';'.join(x))

        logging.info(f'Chunk {chunk_idx}: empty MeshHeadings count = {empty_mesh_count}')
        logging.info(f'Chunk {chunk_idx}: language distribution = {language_counts.to_dict()}')

        return df_filtered[['PMID','ArticleTitle','MeshHeadings', 'AbstractText','NumberOfReferences','CitationSubset','References']]
    except Exception as e:
        logging.error(f'Error processing chunk {chunk_idx}: {e}')
        return pd.DataFrame(columns=['PMID','ArticleTitle','MeshHeadings', 'AbstractText','NumberOfReferences','CitationSubset','References'])

def store_to_csv(df, chunk_idx):
    try:
        with open(output_file, 'a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)
            logging.info(f'Chunk {chunk_idx} stored to CSV')
    except Exception as e:
        logging.error(f'Error storing chunk {chunk_idx} to CSV: {e}')

def process_chunk_wrapper(args):
    return process_chunk(*args)

def main():
    num_cores = max(1, mp.cpu_count() - 5)  # Ensure at least one core is used
    print(f'Number of cores: {num_cores}')

    if not os.path.exists(file_path):
        logging.error(f'File not found: {file_path}')
        return

    with pd.read_csv(file_path, chunksize=10000) as reader:
        for chunk_idx, chunk in enumerate(reader):
            print(f'Processing chunk {chunk_idx}')
            chunks = np.array_split(chunk, num_cores)
            args = [(chunk, chunk_idx) for chunk in chunks]
            with mp.Pool(num_cores) as pool:
                results = pool.map(process_chunk_wrapper, args)

            combined_results = pd.concat(results)
            # store_to_csv(combined_results, chunk_idx)

if __name__ == "__main__":
    main()
