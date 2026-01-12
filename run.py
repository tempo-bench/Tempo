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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate retrieval models on the TEMPO benchmark.")
    parser.add_argument('--task', type=str, required=True, choices=TEMPO_DOMAINS,
                        help='The domain (task) from the TEMPO dataset to evaluate.')
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

    # Setup output directory
    output_dir_path = os.path.join(args.output_dir, args.model, args.task)
    if args.reasoning:
        output_dir_path = os.path.join(output_dir_path, args.reasoning)
    os.makedirs(output_dir_path, exist_ok=True)
    
    score_file_path = os.path.join(output_dir_path, 'scores.json')
    results_file_path = os.path.join(output_dir_path, 'results.json')

    # Load queries from the 'examples' or 'reasoning' split
    print(f"Loading queries for task: {args.task}")
    if args.reasoning:
        print(f"Using reasoning data: {args.reasoning}")
        examples = load_dataset('tempo26/Tempo', args.reasoning, split=args.task, cache_dir=args.cache_dir)
    else:
        examples = load_dataset('tempo26/Tempo', 'examples', split=args.task, cache_dir=args.cache_dir)
    
    # Load documents for the corpus
    print(f"Loading documents for task: {args.task}")
    doc_pairs = load_dataset('tempo26/Tempo', 'documents', split=args.task, cache_dir=args.cache_dir)
    
    doc_ids = [dp['id'] for dp in doc_pairs]
    documents = [dp['content'] for dp in doc_pairs]

    if not os.path.isfile(score_file_path) or args.ignore_cache:
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

        # Prepare queries and ground truth
        queries = []
        query_ids = []
        excluded_ids = {}
        for e in examples:
            queries.append(e["query"])
            query_ids.append(e['id'])
            excluded_ids[e['id']] = e['excluded_ids']
            # Verification
            overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
            assert len(overlap) == 0, f"Query {e['id']} has overlap between gold and excluded IDs."

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
            task=args.task,
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
    print(f"Calculating metrics for {args.task}...")
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    
    print(f"Saving results to {results_file_path}")
    with open(results_file_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"EVALUATION COMPLETE FOR: {args.task.upper()} with model {args.model.upper()}")
    print("="*80)
    print(json.dumps(results, indent=2))
