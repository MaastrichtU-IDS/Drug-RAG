import os
import json
import pandas as pd
from datetime import datetime
from rag_config.run import *
from rag_config.eval import evaluate_result_all
from rag_config.data_loader import load_queries
from rag_config.llm_ import load_llm
import argparse

def run_llm_test(llm_id, queries, retrirvers, output_dir):
    """Run tests for a specific LLM and collect results"""
    print(f"\n{'='*50}")
    print(f"Testing LLM: {llm_id}")
    print(f"{'='*50}\n")
    
    model = load_llm(model=llm_id)
    all_responses = {}
    eval_dataset = []
    
    for query, answer, explanation in queries:
        try:
            print(f"\nQuery: {query}")
            query = query.strip().lower()
            
            # Get contexts from both retrievers
            try:
                pubmed_contexts = retrirvers[1].invoke(query)
                pubmed_contexts = pubmed_contexts if pubmed_contexts else []
            except Exception as e:
                print(f"PubMed retrieval error: {e}")
                pubmed_contexts = []
            
            # Combine contexts
            context = [context.page_content for context in pubmed_contexts]
            dense_retriever_content = retrirvers[0].invoke(query)
            context += [context.page_content for context in dense_retriever_content]
            print(f"Total Retrieved Results: {len(context)}")
            
            # Make multiple attempts for robustness
            claims = []
            responses = []
            for i in range(3):
                response = ask_llm(query, context, model)
                responses.append(response)
                
                # Classify response
                if 'true' in response.lower():
                    claims.append(1)
                    claim_from_explanation = "Claims is True"
                elif 'false' in response.lower():
                    claims.append(0)
                    claim_from_explanation = "Claims is False"
                else:
                    claims.append('nan')
                    claim_from_explanation = "Claims is not substantiated"
                
                print(f"Attempt {i+1} Claim: {claims[i]}")
                
                # Add to evaluation dataset
                eval_dataset.append({
                    'question': query,
                    'contexts': context,
                    'answer': claim_from_explanation,
                    'ground_truth': f"Claims is {answer} based on the explanation: {explanation}",
                })
            
            # Store responses
            all_responses[query] = {
                "claims": claims,
                "explanations": responses
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, llm_id, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save raw responses
    with open(os.path.join(results_dir, 'responses.json'), 'w') as f:
        json.dump(all_responses, f, indent=2)
    
    # Run evaluation metrics
    try:
        eval_results = evaluate_result_all(eval_dataset)
        
        # Save evaluation results
        eval_file = os.path.join(results_dir, 'evaluation_metrics.txt')
        if hasattr(eval_results, "_scores_dict"):
            pd.DataFrame(eval_results._scores_dict.items(), 
                        columns=['Metric', 'Score']).to_csv(eval_file, index=False)
        else:
            with open(eval_file, 'w') as f:
                f.write(str(eval_results))
                
    except Exception as e:
        print(f"Error in evaluation: {e}")
    
    return all_responses, eval_results

def main():
    parser = argparse.ArgumentParser(description='Run Drug Discovery RAG tests across multiple LLMs')
    parser.add_argument('--data_path', type=str, 
                       default='/app/workspace/data/output/abstracts.csv',
                       help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, 
                       default='test_results',
                       help='Directory to store test results')
    parser.add_argument('--recreate_index', action='store_true',
                       help='Recreate the vector index')
    parser.add_argument('--llms', nargs='+',
                       default=['gpt4', 'gpt3.5', 'mixtral'],
                       help='List of LLM IDs to test')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load queries
    queries = load_queries()
    print(f"Total queries loaded: {len(queries)}")
    
    # Initialize retrievers
    retrirvers = create_index(csv_file=args.data_path, recreate=args.recreate_index)
    
    # Run tests for each LLM
    all_results = {}
    for llm_id in args.llms:
        try:
            responses, eval_metrics = run_llm_test(llm_id, queries, retrirvers, args.output_dir)
            all_results[llm_id] = {
                'responses': responses,
                'metrics': eval_metrics
            }
        except Exception as e:
            print(f"Error testing {llm_id}: {e}")
            continue
    
    # Generate comparative analysis
    try:
        comparison_df = pd.DataFrame()
        for llm_id, results in all_results.items():
            if hasattr(results['metrics'], "_scores_dict"):
                metrics = results['metrics']._scores_dict
                comparison_df[llm_id] = pd.Series(metrics)
        
        if not comparison_df.empty:
            comparison_file = os.path.join(args.output_dir, 'llm_comparison.csv')
            comparison_df.to_csv(comparison_file)
            print(f"\nComparative analysis saved to: {comparison_file}")
    except Exception as e:
        print(f"Error generating comparison: {e}")

if __name__ == "__main__":
    main()