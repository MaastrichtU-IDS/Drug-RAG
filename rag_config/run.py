"""
Drug Discovery RAG Pipeline Runner

This script implements the main execution pipeline for drug efficacy evaluation using
Retrieval-Augmented Generation (RAG). It coordinates the retrieval of medical literature,
LLM-based analysis, and evaluation of drug efficacy claims.

Key Components:
- Dual retrieval system (Dense + PubMed)
- LLM-based claim verification
- Evaluation metrics collection
"""

# Import libraries
from indexing import *
from llm_ import load_llm, ask_llm
from data_loader import load_queries
from langchain.schema import Document
from params import MODEL_ID, LLM_ID
import json
from eval import *
import argparse

def pretty_print(contexts):
   """
   Utility function to print context documents in a readable format
   Parameters:
            contexts (list): list of context documents
   """
   for context in contexts:
       print(f"------{context.page_content}----")

#run main function
if __name__ == "__main__":
"""
    Parses command-line arguments for configuration including data paths, 
    model identifiers (for embedding and LLM), collection name, and flags 
    to indicate whether the database should be recreated. Then loads queries 
    to be used later in the pipeline.
"""
    parser = argparse.ArgumentParser(description='Run Drug Discovery RAG')
    # Path to input data (abstracts.csv by default)
    parser.add_argument('--data_path',type=str, help='Path to the data directory', default='/app/workspace/data/output/abstracts.csv')
    #Flag to indicate if database recreation is needed
    parser.add_argument('--recreate', action='store_true', help='Recreate the database')
    # Model and LLM identifiers
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Embedding model id')

    parser.add_argument('--llm_id', type=str, default=LLM_ID, help='LLM model id')
    # Name of the Chroma or vector collection for embeddings storage
    parser.add_argument('--collection_name', type=str, default='pubmed_drugs_medcpt', help='LLM model id')  # pubmed_drugs_medcpt for medcpt embedding model
    # processes the arguments you pass via command-line when executing Python script.
    args = parser.parse_args()
    # Load queries to run the retrieval-augmented generation process
    queries =load_queries()
    print(f"total queries: {len(queries)}")
    # random.shuffle(queries)

    # Initialize retrieval system
    retrirvers = create_index(csv_file=args.data_path,recreate=args.recreate, model_id=args.model_id, collection_name=args.collection_name)
    dense_retrirver = retrirvers[0]
    pubmed_retriever = retrirvers[1]

    # Load LLM model
    llm_id = args.llm_id if args.llm_id else LLM_ID
    model = load_llm(model=llm_id)
    all_responses = {}
    # print(f"Queries: {queries[:5]}")
    eval_dataset = []
    try:
        for query, answer, explanation in queries[:1]:
            try:
                print(f"Query: {query}")
                query = query.strip().lower()
                try:
                    pubmed_contexts = pubmed_retriever.invoke(query)
                    if not pubmed_contexts:
                        pubmed_contexts = []
                except Exception as e:
                    print(f"Error: {e}")
                    pubmed_contexts = []
                    raise e
                print(f"********total pubmed API answers*********:{len(pubmed_contexts)}")
                context = [context.page_content for context in pubmed_contexts]
                dense_retriever_content = dense_retrirver.invoke(query)
                context += [context.page_content for context in dense_retriever_content]
                print(f"********total Retrieved Results*********:{len(context)}")
                print(type(context[0]))
                claims = []
                responses = []
                for i in range(3):
                    response = ask_llm(query, context, model)
                    claim_from_explanation = ""
                    responses.append(response)
                    if 'true' in response.lower():
                        claims.append(1)
                        claim_from_explanation = "Claims is True"
                    elif 'false' in response.lower():
                        claims.append(0) # append 0 if the claim is false
                        claim_from_explanation = "Claims is False"
                    else:
                        claims.append('nan') #append nan if the claim cannot be substantiated
                        claim_from_explanation = "Claims is not substantiated"
                    #print claim and explanation
                    print(f"Claim: {claims[i]}")


                    eval_dataset.append({
                        'question': query,
                        # 'contexts': [k for k in contexts if isinstance(k, str) else k.page_content],
                        'contexts': context,
                        'answer': claim_from_explanation,
                        'ground_truth': f"Claims is {answer} based on the explanation: {explanation}",
                    })
                all_responses.setdefault(query, []).append({
                    "claim":claims,
                    "explanation":responses
                    })  # Correctly handles new or existing keys
            except Exception as e:
                print(f"Error in processing query  : {e}")
                claims.append('nan')
                raise e

        # print(f"Eval dataset: {eval_dataset}")
   
        file_name = f"/app/workspace/data/output/{llm_id}_responses.json"
        with open(file_name, 'w') as f:
            json.dump(all_responses, f)
        
        for k in eval_dataset[0].keys():
            # print(f"Key: {k}")
            # print(f"Type: {type(eval_dataset[0][k])}\n\n")
            if isinstance(eval_dataset[0][k], list):
                for j in eval_dataset[0][k]:
                    print(f"Type of list: {type(j)}\n\n")
                    #if isinstance(j, langchain_core.documents.base.Document):
                    if isinstance(j, Document):
                        print(f"{j.json}\n\n")
                    else:
                        print(j)
                # print(f"Type of list: {type(eval_dataset[0][k][0])}\n\n")
                # print(f"First element: {eval_dataset[0][k][0]}{dir(eval_dataset[0][k][0])}\n\n")

        # with open('/app/workspace/data/output/eval_dataset.json', 'w') as f:
        #     print(dir)
            
        evaluate_result(eval_dataset)

    except Exception as e:
        print(f"Error in query for loop: {e}")
        raise e