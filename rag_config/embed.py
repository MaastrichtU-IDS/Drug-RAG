
from langchain_core.embeddings import Embeddings
import os
import torch
from typing import List
from tqdm import tqdm
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


class BioLMTextEmbedding(Embeddings):
    """
    A class for creating embeddings using a specified transformer model based on Hugging Face's implementations.
    cambridgeltl/SapBERT-from-PubMedBERT-fulltext or xlreator/biosyn-biobert-snomed"""
    def __init__(self, model_id: str = 'stanford-crfm/BioMedLM', device: str = None, **kwargs):
        """
        Initializes the Custom embedding class by loading the specified transformer model.

        Parameters:
            model_id (str): The model identifier from Hugging Face's transformer models.
            device (str, optional): The device to run the model on ("cuda" or "cpu"). 
                                    Defaults to automatically choosing CUDA if available.
        """
        self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=True, truncation=True, padding=True)   
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/app/workspace/resources/models")
        self.model  = self.model.to(self.device) 
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents. For documents with synonyms, the embeddings of the entities and their synonyms are averaged. If only the entity is present, its embedding is used as is."""
        embeddings = []
        pbar = tqdm(total=len(texts), desc="Embedding Documents", unit="doc")
        print(f"embed documents")
        for text in texts:
            entity_embedding = self.embed_query(text)  # Do not squeeze here unless you're sure of the dimensions
            embeddings.append(entity_embedding)  # Convert tensor to list
            pbar.update(1)

        pbar.close()
        return embeddings
    
    def embed_query(self, text: str) -> torch.Tensor:
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        # Get the embeddings (last hidden state)
        outputs = self.model(input_ids, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]  # Access the last hidden state 
        print(f"Embedding dimension: {embeddings.shape}")
        embeddings = embeddings.mean(dim=1).squeeze().cpu().tolist()
        return embeddings

   

# class MedCPTTextEmbedding(Embeddings):
#     def __init__(self, model_name="ncbi/MedCPT-Article-Encoder", device=None):
#         self.model = AutoModel.from_pretrained(model_name, cache_dir="/app/workspace/resources/models")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
#         self.model = self.model.to(self.device)
#         self.model.eval()
#     def parse(self, texts: str) -> str:
#         # title: {title}, abstract: {abstract}
#         articles = []
#         for text in texts:
#             if 'title:' not in text:
#                 articles.append([text, text])
#             else:
#                 print(f"Text: {text}")
#                 items = text.split(",abstract:")
#                 if len(items) == 1:
#                     continue
#                 items = [items[0].split(":")[1].strip().lower(), items[1].strip().lower()]
#                 articles.append(items)
#         return articles
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         try:
#             texts = self.parse(texts)
#             embeddings = []
#             with torch.no_grad():
#                 # tokenize the queries
#                 encoded = self.tokenizer(
#                     texts, 
#                     truncation=True, 
#                     padding=True, 
#                     return_tensors='pt', 
#                     max_length=512,
#                 ).to(self.device)
                
#                 # encode the queries (use the [CLS] last hidden states as the representations)
#                 embeds = self.model(**encoded).last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
#                 print(f"generated embeddings: {len(embeds)}")
#                 embeddings.append(embeds)
#             return embeddings
#         except Exception as e:
#             raise e
    
#     def embed_query(self, text: str) -> List[float]:
#         queries = [text]
#         with torch.no_grad():
#             # tokenize the queries
#             encoded = self.tokenizer(
#                 queries, 
#                 truncation=True, 
#                 padding=True, 
#                 return_tensors='pt', 
#                 max_length=64,
#             ).to(self.device)
            
#             # encode the queries (use the [CLS] last hidden states as the representations)
#         embeds = self.model(**encoded).last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
#         print(f"length of embeddings: {len(embeds)}")
#         return embeds
    


class MedCPTTextEmbedding(Embeddings):
    def __init__(self, model_name="ncbi/MedCPT-Article-Encoder", device=None):
        self.model = AutoModel.from_pretrained(model_name, cache_dir="/app/workspace/resources/models")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def parse(self, texts: List[str]) -> List[List[str]]:
        articles = []
        for text in texts:
            if 'title:' not in text:
                # If there's no title, use the same text for both title and abstract
                articles.append([text, text])
            else:
                # Parse the title and abstract
                items = text.split(",abstract:")
                if len(items) == 1:
                    continue
                title = items[0].split(":")[1].strip()
                abstract = items[1].strip()
                articles.append([title, abstract])
        return articles

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            # Parse the input texts into the required format (list of [title, abstract])
            articles = self.parse(texts)
            with torch.no_grad():
                # Tokenize the articles
                encoded = self.tokenizer(
                    articles,
                    truncation=True,
                    padding=True,
                    return_tensors='pt',
                    max_length=512,
                ).to(self.device)

                # Encode the articles (use the [CLS] last hidden states as the representations)
                embeds = self.model(**encoded).last_hidden_state[:, 0, :].detach().cpu().numpy()
            
            # Convert NumPy array to list of lists of floats for Qdrant
            embeds = embeds.tolist()
            print(f"length of embeddings in document: {len(embeds[0])}")
            return embeds
        except Exception as e:
            raise e

    def embed_query(self, text: str) -> List[float]:
        try:
            queries = [text]  # Queries must be structured as [title, abstract] (using the same text if no abstract is provided)
            with torch.no_grad():
                # Tokenize the queries
                encoded = self.tokenizer(
                    queries, 
                    truncation=True, 
                    padding=True, 
                    return_tensors='pt', 
                    max_length=64,
                ).to(self.device)
                
                # Encode the queries (use the [CLS] last hidden states as the representations)
                embeds = self.model(**encoded).last_hidden_state[:, 0, :].detach().cpu().numpy()
            print(f"length of embeddings in query: {len(embeds[0].tolist())}")
            # Convert NumPy array to list of floats for Qdrant
            return embeds[0].tolist()
        except Exception as e:
            raise e
