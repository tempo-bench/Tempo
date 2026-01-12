import os
import argparse
import json
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS, calculate_retrieval_metrics
from datasets import load_dataset

# Define the domains available in the TEMPO dataset
TEMPO_DOMAINS = [
    'bitcoin', 'cardano', 'economics', 'genealogy', 'history', 'hsm', 
    'iota', 'law', 'monero', 'politics', 'quant', 'travel', 'workplace'
]

def evaluate_task(task, args, config):
    print(f"\n{'='*80}")
    print(f"EVALUATING TASK: {task.upper()}")
    print(f"{'='*80}")

    # Setup output directory for this task
    output_dir_path = os.path.join(args.output_dir, args.model, task)
    if args.reasoning:
        output_dir_path = os.path.join(output_dir_path, args.reasoning)
    os.makedirs(output_dir_path, exist_ok=True)
    
    score_file_path = os.path.join(output_dir_path, 'scores.json')
    results_file_path = os.path.join(output_dir_path, 'results.json')

    # Load queries
    print(f"Loading queries for task: {task}")
    try:
        if args.reasoning:
            print(f"Using reasoning data: {args.reasoning}")
            examples = load_dataset('tempo26/Tempo', args.reasoning, split=task, cache_dir=args.cache_dir)
        else:
            examples = load_dataset('tempo26/Tempo', 'examples', split=task, cache_dir=args.cache_dir)
    except Exception as e:
        print(f"Error loading queries for {task}: {e}")
        return None
    
    # Load documents
    print(f"Loading documents for task: {task}")
    try:
        doc_pairs = load_dataset('tempo26/Tempo', 'documents', split=task, cache_dir=args.cache_dir)
    except Exception as e:
        print(f"Error loading documents for {task}: {e}")
        return None
    
    doc_ids = [dp['id'] for dp in doc_pairs]
    documents = [dp['content'] for dp in doc_pairs]

    if not os.path.isfile(score_file_path) or args.ignore_cache:
        # Prepare queries and excluded_ids
        queries = []
        query_ids = []
        excluded_ids = {} # Initialize
        
        for e in examples:
            queries.append(e["query"])
            query_ids.append(e['id'])
            # Check for negative_ids in the dataset item, use it if present
            if 'negative_ids' in e:
                 excluded_ids[e['id']] = e['negative_ids']
            else:
                 excluded_ids[e['id']] = []

        assert len(queries) == len(query_ids)
        assert len(doc_ids) == len(documents)

        print(f"Total queries: {len(queries)}")
        print(f"Total documents: {len(documents)}")

        if args.debug:
            print("[DEBUG MODE] Using a subset of 30 documents.")
            documents = documents[:30]
            doc_ids = doc_ids[:30]

        # Prepare retrieval function arguments
        kwargs = {}
        if args.query_max_length > 0:
            kwargs['query_max_length'] = args.query_max_length
        if args.doc_max_length > 0:
            kwargs['doc_max_length'] = args.doc_max_length
        if args.encode_batch_size > 0:
            kwargs['batch_size'] = args.encode_batch_size
        if args.key is not None:
            kwargs['key'] = args.key
        if args.ignore_cache:
            kwargs['ignore_cache'] = args.ignore_cache
        if args.checkpoint:
            kwargs['checkpoint'] = args.checkpoint

        # Run retrieval
        print(f"Running retrieval with model: {args.model}")
        scores = RETRIEVAL_FUNCS[args.model](
            queries=queries,
            query_ids=query_ids,
            documents=documents,
            excluded_ids=excluded_ids,
            instructions=config['instructions'],
            doc_ids=doc_ids,
            task=task,
            cache_dir=args.cache_dir,
            long_context=False, # TEMPO does not have a long_context split
            model_id=args.model,
            **kwargs
        )
        
        print(f"Saving scores to {score_file_path}")
        with open(score_file_path, 'w') as f:
            json.dump(scores, f, indent=2)
    else:
        print(f"Loading existing scores from {score_file_path}")
        with open(score_file_path) as f:
            scores = json.load(f)

    # Prepare ground truth for evaluation
    ground_truth = {}
    for e in tqdm(examples, desc="Preparing ground truth"):
        ground_truth[e['id']] = {gid: 1 for gid in e['gold_ids']}

    # Calculate and save metrics
    print(f"Calculating metrics for {task}...")
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    
    print(f"Saving results to {results_file_path}")
    with open(results_file_path, 'w') as f:
        json.dump(results, f, indent=2)

    return {
        'task': task,
        'num_queries': len(examples),
        'results': results
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate retrieval models on the TEMPO benchmark.")
    parser.add_argument('--task', type=str, choices=TEMPO_DOMAINS + ['all'], default='all',
                        help='The domain (task) from the TEMPO dataset to evaluate. Use "all" to run all domains.')
    parser.add_argument('--model', type=str, required=True, choices=[
        'sf', 'qwen', 'qwen2', 'e5', 'bm25', 'sbert', 'bge', 'inst-l', 'inst-xl', 
        'grit', 'cohere', 'voyage', 'openai', 'google', 'diver-retriever', 
        'contriever', 'reasonir', 'm2', 'rader', 'nomic'
    ], help='The retrieval model to use.')
    parser.add_argument('--reasoning', type=str, default=None,
                        choices=['qwen72b_reason', 'gpt4o_reason', 'deepseek_reason', 'llama70b_reason'],
                        help='Optional reasoning chain to prepend to the query.')
    
    # Model and retrieval configuration
    parser.add_argument('--query_max_length', type=int, default=-1, help='Max length for query encoding.')
    parser.add_argument('--doc_max_length', type=int, default=-1, help='Max length for document encoding.')
    parser.add_argument('--encode_batch_size', type=int, default=-1, help='Batch size for encoding.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a model checkpoint.')
    parser.add_argument('--key', type=str, default=None, help='API key for retrieval services if needed.')

    # Caching and output directories
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs.')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Directory for caching embeddings.')
    parser.add_argument('--config_dir', type=str, default='configs', help='Directory for model configurations.')
    
    # Flags
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with a small subset of data.')
    parser.add_argument('--ignore_cache', action='store_true', help='Ignore and overwrite existing cache.')
    args = parser.parse_args()

    # Load model-specific instruction config
    config_path = os.path.join(args.config_dir, args.model, "default.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Default instructions if no config file is found
        config = {
            'instructions': {
                'query': 'Represent this query for retrieving relevant documents: ',
                'document': ''
            }
        }

    # Determine which tasks to run
    if args.task == 'all':
        tasks_to_run = TEMPO_DOMAINS
    else:
        tasks_to_run = [args.task]

    all_task_results = {}
    
    for task in tasks_to_run:
        result = evaluate_task(task, args, config)
        if result:
            all_task_results[task] = result

    # Save aggregated results (summary.json)
    if not args.reasoning:
        model_output_dir = os.path.join(args.output_dir, args.model)
    else:
        model_output_dir = os.path.join(args.output_dir, args.model, args.reasoning)
        
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print(f"{'='*80}")

    if len(all_task_results) > 0:
        # Get all metric keys from first task
        first_task = list(all_task_results.values())[0]
        metric_keys = list(first_task['results'].keys())
        
        aggregated_metrics = {}
        for key in metric_keys:
            values = [result['results'][key] for result in all_task_results.values()]
            aggregated_metrics[key] = round(sum(values) / len(values), 5)
        
        print("\nAverage across all tasks:")
        print(json.dumps(aggregated_metrics, indent=2))
        
        summary = {
            'model': args.model,
            'reasoning': args.reasoning,
            'num_tasks': len(all_task_results),
            'tasks': list(all_task_results.keys()),
            'aggregated_metrics': aggregated_metrics,
            'per_task': all_task_results
        }
        
        summary_path = os.path.join(model_output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Evaluation complete! Results saved to {model_output_dir}")
        print(f"  - Per-task results: {model_output_dir}/<task>/results.json")
        print(f"  - Aggregated summary: {summary_path}")
    else:
        print("\nNo tasks were successfully evaluated.")
