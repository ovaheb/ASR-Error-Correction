from codes.utils import *
from codes.ec_methods import *


async def track_progress(tasks):
    """Tracks progress while tasks are running."""
    
    total_tasks = len(tasks)
    while not all(task.done() for task in tasks):
        completed = sum(task.done() for task in tasks)
        print(f"Progress: {completed}/{total_tasks} tests completed!", end="\r")
        await asyncio.sleep(0.1)
    print(f"Progress: Batch of {total_tasks} tests completed!", flush=True)
    return await asyncio.gather(*tasks)

    
async def process_batch(dataset: Dataset, indices: List[int], model: str, client: openai.AsyncOpenAI, postprocessing: Callable[[List[str]], str], generation_config: dict, few_shot: int, error_examples: List[str]) -> List[str]:
    """Processes the dataset asynchronously using OpenAI API with progress tracking."""
    
    tasks = []
    for idx in indices:
        hypotheses, reference = extract_hypotheses(dataset, idx)
        if inspect.iscoroutinefunction(postprocessing):
            if few_shot==0:
                tasks.append(asyncio.create_task(postprocessing(hypotheses, client, model, generation_config)))
            else:
                tasks.append(asyncio.create_task(postprocessing(hypotheses, client, model, generation_config, few_shot, error_examples)))
        else:
            tasks.append(asyncio.create_task(asyncio.to_thread(postprocessing, hypotheses, reference)))
    
    results = await track_progress(tasks)
    return results
    
async def evaluate_model_parallel(dataset: Dataset, model: str, client: openai.AsyncOpenAI, postprocessing: Callable[[List[str]], str], generation_config: dict, results_path: str, step: int=256, experimental=False, few_shot: int=0, error_examples: List[str]=None):
    """Evaluates the model asynchronously with progress tracking, handling Jupyter compatibility."""
    
    total_rows = len(dataset)
    all_predictions = []
    for start in range(0, total_rows, step):
        end = min(start + step, total_rows)
        batch_indices = list(range(start, end))
        batch_predictions = await process_batch(dataset, batch_indices, model, client, postprocessing, generation_config, few_shot, error_examples)
        all_predictions.extend(batch_predictions)
    
    # Normalize for evaluation
    if 'DeepSeek' in model:
        all_predictions = [clean_deepseek_output(pred) for pred in all_predictions] 
    all_predictions = [clean_asr_output(remove_punctuation(pred.lower())) for pred in all_predictions]
    all_references = dataset['target'] if 'target' in dataset.features else dataset['output']
    all_references = [clean_asr_output(remove_punctuation(ref.lower())) for ref in all_references]

    # Print 3 random results for manual review
    random_indices = random.sample(range(len(all_predictions)), 3)
    print("-" * 100)
    for idx in random_indices:
        print(f"Sample {idx + 1}")
        report = jiwer.process_words(all_references[idx], all_predictions[idx])
        print(jiwer.visualize_alignment(report, show_measures=False))
        print("-" * 100)
        
    if not experimental:
        save_results(dataset, all_predictions, model, postprocessing.__name__, results_path)
        
    # Compute evaluation metrics
    wer_scores = np.array([jiwer.wer(ref, pred) for ref, pred in zip(all_references, all_predictions)])
    bertscore = compute_bertscore(all_predictions, all_references)
    metrics = {
        'WER': round(wer_scores.mean().item(), 3),
        'METEOR': round(compute_meteor(all_predictions, all_references), 3),
        'BERT Precision': round(bertscore['precision'], 3),
        'BERT Recall': round(bertscore['recall'], 3),
        'BERT F1': round(bertscore['f1'], 3),
    }
    return metrics


async def run_evaluation(dataset, model, client, generation_config, results_path, disable_zsun=False, disable_zsco=False, disable_zscl=False):
    metrics_zero_shot_unconstrained = None
    metrics_zero_shot_constrained = None
    metrics_zero_shot_closest = None
    if not disable_zsun:
        print("Evaluating Zero-shot Unconstrained:")
        metrics_zero_shot_unconstrained = await evaluate_model_parallel(dataset, model, client, zero_shot_unconstrained, generation_config, results_path)
    
    if not disable_zsco:
        print("Evaluating Zero-shot Constrained:")
        metrics_zero_shot_constrained = await evaluate_model_parallel(dataset, model, client, zero_shot_constrained, generation_config, results_path)
    
    if not disable_zscl:
        print("Evaluating Zero-shot Closest:")
        metrics_zero_shot_closest = await evaluate_model_parallel(dataset, model, client, zero_shot_closest, generation_config, results_path)
    
    print("Evaluating Oracle:")
    metrics_get_oracle_hypothesis = await evaluate_model_parallel(dataset, model, client, get_oracle_hypothesis, generation_config, results_path)
    
    print("Evaluating Top 1:")
    metrics_get_top1_hypothesis = await evaluate_model_parallel(dataset, model, client, get_top1_hypothesis, generation_config, results_path)

    results_table = {
        "Top 1": metrics_get_top1_hypothesis,
        "Oracle": metrics_get_oracle_hypothesis,
    }

    if metrics_zero_shot_unconstrained is not None:
        results_table["Zero-shot Uncon"] = metrics_zero_shot_unconstrained
    if metrics_zero_shot_constrained is not None:
        results_table["Zero-shot Constr"] = metrics_zero_shot_constrained
    if metrics_zero_shot_closest is not None:
        results_table["Zero-shot Closest"] = metrics_zero_shot_closest

    results_table = pd.DataFrame.from_dict(results_table, orient='index')
    results_table = results_table[['WER', 'METEOR', 'BERT Precision', 'BERT Recall', 'BERT F1']]

    # Save as JSON
    csv_path = results_path.replace(".json", f"_{model}.csv")
    results_table.to_csv(csv_path)
    print(f"Benchmark saved to {csv_path}")
    return results_table



if __name__ == "__main__":
    txt = "Simple test!"
    print((txt))