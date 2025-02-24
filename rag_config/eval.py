# import metrics
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics import context_precision, faithfulness, summarization_score
from llm_ import load_llm
from typing import List, Dict
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import evaluate
from langchain_community.embeddings.ollama import OllamaEmbeddings
from datasets import Dataset
from datasets import DatasetDict
from langchain.callbacks.tracers import LangChainTracer
import os

tracer = LangChainTracer(project_name="callback-experiments")
#os.environ["OPENAI_API_KEY"] = " "

def evaluate_result(result):
    try:
        dataset_dict = DatasetDict({
            "eval": Dataset.from_list(result)
        })
        # dataset = Dataset.from_list(result, split="eval")
        # print(f"type of dataset: {type(dataset)}")
        evaluator_llm = LangchainLLMWrapper(load_llm())
        # llama_embed = OllamaEmbeddings(
        #      base_url="http://ollama:11434",  # Ollama server endpoint
        #                 model="llama3.1:8b",
        #                 temperature=0,
        # )
        results = evaluate(dataset_dict["eval"],metrics=[context_precision],callbacks=[tracer], llm=evaluator_llm)
        # result is Result object with Dataset
        print(f"Type of results: {type(results)}")
        print(f"Results  content: {results}")
        print(f"Results dir: {dir(results)}")
        #save the results to a file
        with open('/app/workspace/data/output/eval_results.txt', 'w') as f:
             if hasattr(results, "items"):
                for metric, score in results.items():
                    f.write(f"{metric}: {score}\n")
             elif hasattr(results, "to_dict"):  # Check if there's a dict conversion method
                results_dict = results.to_dict()
                for metric, score in results_dict.items():
                    f.write(f"{metric}: {score}\n")
             else:
                # Fallback: direct print if it's a non-standard structure
                f.write(f"Unexpected results format: {results}")
       #with open('/app/workspace/data/output/eval_results.txt', 'w') as f:
       #     for res in results:
       #         f.write(f"{res}\n")
        return results
    except Exception as e:
        print(f"Error in evalute results: {e}")
        raise e
    

def evaluate_result_all(result):
    try:
        dataset_dict = DatasetDict({
            "eval": Dataset.from_list(result)
        })
        
        evaluator_llm = LangchainLLMWrapper(load_llm())

        # Define metrics to be used in evaluation
        results = evaluate(
            dataset_dict["eval"],
            metrics=[context_precision, faithfulness, summarization_score],
            callbacks=[tracer],
            llm=evaluator_llm
        )
        
        # Debugging information
        print(f"Type of results: {type(results)}")
        print(f"Results content: {results}")
        print(f"Results dir: {dir(results)}")

        # Save the results to a file
        with open('/app/workspace/data/output/eval_results.txt', 'w') as f:
            if hasattr(results, "_scores_dict"):
                for metric, score in results._scores_dict.items():
                    f.write(f"{metric}: {score}\n")
            elif hasattr(results, "to_pandas"):
                results_df = results.to_pandas()
                f.write(results_df.to_string(index=False))
            else:
                f.write(f"Unexpected results format: {results}")
                
        return results
    except Exception as e:
        print(f"Error in evaluate results: {e}")
        raise e