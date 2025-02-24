from langchain_community.retrievers import PubMedRetriever
from qdrant_client import QdrantClient
from langchain_qdrant.qdrant import QdrantVectorStore
from langchain_qdrant.vectorstores import Qdrant
# from langchain_community.document_loaders import TextLoader
from data_loader import *
from embed import *
from langchain_text_splitters.base import TokenTextSplitter
from params import VECTOR_PATH, QDRANT_PORT, MODEL_ID,COLLECTION_NAME
import qdrant_client.http.models as rest

def create_index(csv_file='/app/workspace/data/output/abstracts.csv', model_id:str=MODEL_ID,
                  recreate=False, url=VECTOR_PATH, port=QDRANT_PORT, collection_name=COLLECTION_NAME):

    embeddings = MedCPTTextEmbedding() if 'medcpt' in model_id.lower() else BioLMTextEmbedding()
    pubmed_retriever = PubMedRetriever(email='komalsyeda29@gmail.com')
    client = QdrantClient(url=url, port=port, https=True)
    # client.delete_collection('drug_discovery')
    collection_exist = client.collection_exists(collection_name)
    if not collection_exist or recreate:
        docs = load_docs(csv_file)
        child_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=30)
        docs =  child_splitter.split_documents(docs)
        vector_store = QdrantVectorStore.from_documents(
            docs,
            embedding=embeddings,
            url=VECTOR_PATH, 
            port=QDRANT_PORT,
            https = True,
            vector_name ='drug_vector',
            collection_name=collection_name,
                        vector_params  = { 
            "size": 2560,
            "distance": rest.Distance.COSINE,
            "hnsw_config": rest.HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                    payload_m = 16 
                ),
            
            "quantization_config":rest.ScalarQuantization(
                    scalar=rest.ScalarQuantizationConfig(
                        type=rest.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            "on_disk":True
            },
            sparse_vector_params = {
                "modifier":rest.Modifier.IDF,
                "index":{
                    "full_scan_threshold":10000,
                    "on_disk":True
                } 
            },
            force_recreate=True    
            
        )
    else:
        print(type(embeddings), embeddings)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            distance=rest.Distance.COSINE,
            vector_name="drug_vector",
            validate_collection_config=True
        )

    qdrant_retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'lambda_mult': 0.25},
                return_source_documents=True,
                    )
    return [qdrant_retriever,pubmed_retriever]
        

#main function
if __name__ == "__main__":
    retrirvers = create_index()
    dense_retrirver = retrirvers[0]
    queries =load_queries()
    for query,_,_ in queries:
        print(f"Query: {query}")
        print(dense_retrirver.invoke(query))