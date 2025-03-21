import pandas as pd
from langchain.schema import Document
from uuid import uuid4
def load_docs(csv_file='data.csv', **kwargs):
    """
    Load documents from a CSV file.
    parameters:
        csv_file(str): path to the CSV file
        kwargs: additional keyword arguments for pandas.read_csv
    returns:
        docs(list): list of Document

    """
    # Load the CSV file, pass additional keyword arguments to pandas.read_csv BY using **kwargs
    df = pd.read_csv(csv_file, **kwargs)
    #create a list of Document objects from the dataframe
    docs = []
    #iterate over the rows of the dataframe to 
    for i, row in df.iterrows():
        doc = Document(
            id = str(uuid4()),
            page_content=f"title:{row['ArticleTitle']},abstract:{row['AbstractText']}",
            metadata={
                "labels": row['MeshHeadings'],
                "reference_count":row['NumberOfReferences'] if pd.notna(row['NumberOfReferences']) else 0,
                "reference":row['References'] if pd.notna(row['References']) else None,
                "title": row['ArticleTitle'],
                'citiation_subset': row['CitationSubset'],
            }
        )
        docs.append(doc)
    return docs


def load_queries(csv_file='/app/workspace/data/input/eval_datasets/cvd_drug_claims_test.csv'):
    """
    Load queries from a text file.
    """
    queries = []
    df = pd.read_csv(csv_file, sep=',',index_col=False)
    # print(df.head())
    for _, row in df.iterrows():
        queries.append((str(row['Question']).lower(), str(row['Answer']).lower(), str(row['Explanation']).lower()))
    return queries