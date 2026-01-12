import os
import argparse
import json
from tqdm import tqdm
from collections import defaultdict
from retrievers import RETRIEVAL_FUNCS, calculate_retrieval_metrics
from datasets import load_dataset

# Define the domains available in the TEMPO dataset
TEMPO_DOMAINS = [
    'bitcoin', 'cardano', 'economics', 'genealogy', 'history', 'hsm', 
    'iota', 'law', 'monero', 'politics', 'quant', 'travel', 'workplace'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate step-wise retrieval on the TEMPO benchmark.")
    parser.add_argument('--task', type=str, required=True, choices=TEMPO_DOMAINS,
                        help='The domain (task) from the TEMPO dataset to evaluate.')
    parser.add_argument('--model', type=str, required=True, choices=list(RETRIEVAL_FUNCS.keys()),
                        help='The retrieval model to use.')
    
    # Model and retrieval configuration
    parser.add_argument('--query_max_length', type=int, default=-1, help='Max length for query encoding.')
    parser.add_argument('--doc_max_length', type=int, default=-1, help='Max length for document encoding.')
    parser.add_argument('--encode_batch_size', type=int, default=-1, help='Batch size for encoding.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a model checkpoint.')
    parser.add_argument('--key', type=str, default=None, help='API key for retrieval services if needed.')

    # Caching and output directories
    parser.add_argument('--output_dir', type=str, default='outputs_steps', help='Directory to save outputs.')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Directory for caching embeddings.')
    parser.add_argument('--config_dir', type=str, default='configs', help='Directory for model configurations.')
    
    # Flags
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with a small subset of data.')
    parser.add_argument('--ignore_cache', action='store_true', help='Ignore and overwrite existing cache.')
    args = parser.parse_args()

    # Setup output directory
    output_dir_path = os.path.join(args.output_dir, args.model, args.task)
    os.makedirs(output_dir_path, exist_ok=True)
    
    results_file_path = os.path.join(output_dir_path, 'results.json')

    # Load documents for the corpus
    print(f"Loading documents for task: {args.task}")
    doc_pairs = load_dataset('tempo26/Tempo', 'documents', split=args.task, cache_dir=args.cache_dir)
    doc_ids = [dp['id'] for dp in doc_pairs]
    documents = [dp['content'] for dp in doc_pairs]

    # Load steps data
    print(f"Loading steps for task: {args.task}")
    steps_data_flat = load_dataset('tempo26/Tempo', 'steps', split=args.task, cache_dir=args.cache_dir)

    # Group steps by query_id
    grouped_steps = defaultdict(list)
    for step in steps_data_flat:
        grouped_steps[step['q_id']].append(step)

    queries_data = []
    for q_id, steps in grouped_steps.items():
        # All steps for a query should have the same base query text and excluded_ids
        base_query = steps[0]['query']
        excluded_ids_list = steps[0]['excluded_ids']
        
        valid_steps = []
        for s in steps:
            if s['gold_ids']: # Ensure step has gold documents
                valid_steps.append({
                    'step_id': s['step_id'],
                    'step_text': s['step'],
                    'gold_ids': s['gold_ids']
                })

        if valid_steps:
            queries_data.append({
                'query_id': q_id,
                'base_query': base_query,
                'steps': valid_steps,
                'excluded_ids': excluded_ids_list
            })
    
    print(f"Loaded {len(queries_data)} queries with a total of {sum(len(q['steps']) for q in queries_data)} steps.")

    if args.debug:
        print("[DEBUG MODE] Using a subset of 30 documents and 5 queries.")
        documents = documents[:30]
        doc_ids = doc_ids[:30]
        queries_data = queries_data[:5]

    # Prepare retrieval function arguments
    kwargs = {}
    if args.query_max_length > 0: kwargs['query_max_length'] = args.query_max_length
    if args.doc_max_length > 0: kwargs['doc_max_length'] = args.doc_max_length
    if args.encode_batch_size > 0: kwargs['batch_size'] = args.encode_batch_size
    if args.key is not None: kwargs['key'] = args.key
    if args.ignore_cache: kwargs['ignore_cache'] = args.ignore_cache
    if args.checkpoint: kwargs['checkpoint'] = args.checkpoint

    # Load model-specific instruction config
    config_path = os.path.join(args.config_dir, args.model, "default.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {'instructions': {'query': 'Represent this query for retrieving relevant documents: ', 'document': ''}}

    # --- BATCH RETRIEVAL LOGIC ---
    
    # 1. Pre-cache document embeddings if not already done
    print(f"\n[CACHE WARMING] Caching document embeddings for {args.model} on task {args.task}...")
    dummy_cache_file = os.path.join(output_dir_path, 'scores_dummy_cache.json')
    if not os.path.isfile(dummy_cache_file) or args.ignore_cache:
        _ = RETRIEVAL_FUNCS[args.model](
            queries=["dummy query to warm cache"], query_ids=["dummy"],
            documents=documents, doc_ids=doc_ids, task=args.task,
            instructions=config['instructions'], excluded_ids={"dummy": []},
            cache_dir=args.cache_dir, long_context=False, model_id=args.model, **kwargs
        )
        with open(dummy_cache_file, 'w') as f: json.dump({"cached": True}, f)
        print("  ✓ Document embeddings cached!")
    else:
        print("  ✓ Document embeddings already cached.")

    # 2. Prepare all steps from all queries for a single batch retrieval call
    all_step_queries = []
    all_step_query_ids = []
    step_id_to_info = {}

    print("\n[PREPARING BATCH] Aggregating all steps for batch processing...")
    for query_data in queries_data:
        for step in query_data['steps']:
            step_query_id = step['step_id']
            score_file_path = os.path.join(output_dir_path, f'scores_{step_query_id}.json')

            if not os.path.isfile(score_file_path) or args.ignore_cache:
                combined_query = f"{query_data['base_query']}\n\nStep: {step['step_text']}"
                all_step_queries.append(combined_query)
                all_step_query_ids.append(step_query_id)
                step_id_to_info[step_query_id] = {
                    'gold_ids': step['gold_ids'],
                    'excluded_ids': query_data['excluded_ids'],
                    'score_file_path': score_file_path
                }
    
    # 3. Execute the batch retrieval if there are any steps to process
    if all_step_queries:
        print(f"\n[BATCH RETRIEVAL] Processing {len(all_step_queries)} steps in a single batch...")
        excluded_ids_batch = {qid: step_id_to_info[qid]['excluded_ids'] for qid in all_step_query_ids}
        
        all_scores = RETRIEVAL_FUNCS[args.model](
            queries=all_step_queries, query_ids=all_step_query_ids,
            documents=documents, doc_ids=doc_ids, task=args.task,
            instructions=config['instructions'], excluded_ids=excluded_ids_batch,
            cache_dir=args.cache_dir, long_context=False, model_id=args.model, **kwargs
        )

        print("\n[SAVING SCORES] Writing individual score files for each step...")
        for step_query_id in tqdm(all_step_query_ids, desc="Saving scores"):
            info = step_id_to_info[step_query_id]
            with open(info['score_file_path'], 'w') as f:
                json.dump({step_query_id: all_scores[step_query_id]}, f, indent=2)
    else:
        print("\n[CACHE HIT] All step scores are already cached.")

    # 4. Calculate metrics for all steps
    print("\n[METRICS] Calculating and aggregating metrics across all steps...")
    step_metrics_accumulator = defaultdict(list)
    total_steps_count = 0

    for query_data in queries_data:
        for step in query_data['steps']:
            step_query_id = step['step_id']
            score_file_path = os.path.join(output_dir_path, f'scores_{step_query_id}.json')
            
            with open(score_file_path) as f:
                scores = json.load(f)
            
            ground_truth = {step_query_id: {gid: 1 for gid in step['gold_ids']}}
            step_results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
            
            for metric, value in step_results.items():
                step_metrics_accumulator[metric].append(value)
            total_steps_count += 1
            
    if total_steps_count > 0:
        averaged_metrics = {
            metric: round(sum(values) / len(values), 5)
            for metric, values in step_metrics_accumulator.items()
        }
        
        # Save aggregated results
        summary_data = {
            'model': args.model,
            'task': args.task,
            'num_queries': len(queries_data),
            'total_steps': total_steps_count,
            'aggregated_metrics': averaged_metrics,
        }
        with open(results_file_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        print("\n" + "="*80)
        print(f"STEP-WISE EVALUATION COMPLETE FOR: {args.task.upper()} with model {args.model.upper()}")
        print("="*80)
        print(json.dumps(summary_data, indent=2))
    else:
        print("\nNo steps were evaluated.")
