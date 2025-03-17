import asyncio
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import List, TYPE_CHECKING



# ADD THIS IMPORT STATEMENT AT THE TOP:
from correction_strategies import (
    OneShotUnconstrainedCorrection,
    OneShotClosestCorrection,
    OracleHypothesisSelection,
    Top1HypothesisSelection
)

if TYPE_CHECKING:
    from llm_client import LLMClient
    from correction_strategies import CorrectionStrategy
    from metrics import MetricsCalculator
    from data_handler import DataHandler
    from utils import ProgressTracker
    
    
    

class EvaluationPipeline:
    """
    Orchestrates the evaluation of different ASR error correction strategies.
    """

    def __init__(self, metrics_calculator: 'MetricsCalculator', data_handler: 'DataHandler', progress_tracker: 'ProgressTracker'):
        """
        Initialize the EvaluationPipeline with necessary utility classes.

        Args:
            metrics_calculator (MetricsCalculator): Instance of MetricsCalculator.
            data_handler (DataHandler): Instance of DataHandler.
            progress_tracker (ProgressTracker): Instance of ProgressTracker.
        """
        self.metrics_calculator = metrics_calculator
        self.data_handler = data_handler
        self.progress_tracker = progress_tracker

    async def process_batch(self, dataset: Dataset, model: str, llm_client: 'LLMClient', correction_strategy: 'CorrectionStrategy',
                            generation_config: dict) -> List[str]:
        """
        Processes a batch of data using a given correction strategy asynchronously.

        Args:
            dataset (Dataset): The dataset to process.
            model (str): The name of the language model.
            llm_client (LLMClient): Instance of LLMClient.
            correction_strategy (CorrectionStrategy): The correction strategy to apply.
            generation_config (dict): Generation configuration for the language model.

        Returns:
            List[str]: List of corrected transcripts.
        """
        tasks = []
        for idx in range(len(dataset)):
            hypotheses, reference = self.data_handler.extract_hypotheses(dataset, idx)
            if isinstance(correction_strategy, OracleHypothesisSelection): # Oracle needs reference
                tasks.append(asyncio.create_task(correction_strategy.correct(hypotheses, llm_client, model, generation_config, reference=reference)))
            else:
                tasks.append(asyncio.create_task(correction_strategy.correct(hypotheses, llm_client, model, generation_config)))

        print("Submitted all tasks!")
        results = await self.progress_tracker.track_progress(tasks)
        return results

    async def evaluate_model_parallel(self, dataset: Dataset, model: str, llm_client: 'LLMClient', correction_strategy: 'CorrectionStrategy',
                                    generation_config: dict, results_path: str):
        """
        Evaluates a correction strategy on the dataset and computes metrics.

        Args:
            dataset (Dataset): The dataset to evaluate on.
            model (str): The name of the language model.
            llm_client (LLMClient): Instance of LLMClient.
            correction_strategy (CorrectionStrategy): The correction strategy to evaluate.
            generation_config (dict): Generation configuration for the language model.
            results_path (str): Path to save the results.

        Returns:
            dict: Dictionary of evaluation metrics.
        """
        all_predictions = await self.process_batch(dataset, model, llm_client, correction_strategy, generation_config)
        all_references = dataset['target'] if 'target' in dataset.features else dataset['output']

        # Normalize for evaluation
        all_predictions = [pred.lower() for pred in all_predictions]
        all_references = [ref.lower() for ref in all_references]

        # Print 3 random results for manual review
        random_indices = random.sample(range(len(all_predictions)), 3)
        for idx in random_indices:
            print(f"Sample {idx + 1}")
            print(f"Target: {all_references[idx]}")
            print(f"Pred:   {all_predictions[idx]}")
            print("-" * 50)

        self.data_handler.save_results(dataset, all_predictions, model, type(correction_strategy).__name__, results_path)

        # Compute evaluation metrics
        wer_scores = np.array([self.metrics_calculator.compute_wer(reference=ref, hypothesis=pred) for ref, pred in zip(all_references, all_predictions)])
        bertscore = self.metrics_calculator.compute_bertscore(all_predictions, all_references)
        metrics = {
            'WER': round(wer_scores.mean().item(), 3),
            'METEOR': round(self.metrics_calculator.compute_meteor(all_predictions, all_references), 3),
            'BERT Precision': round(bertscore['precision'], 3),
            'BERT Recall': round(bertscore['recall'], 3),
            'BERT F1': round(bertscore['f1'], 3),
        }
        return metrics

    async def run_evaluation(self, dataset: Dataset, model: str, llm_client: 'LLMClient', generation_config: dict, results_path: str):
        """
        Runs evaluation for all defined correction strategies.

        Args:
            dataset (Dataset): The dataset to evaluate on.
            model (str): The name of the language model.
            llm_client (LLMClient): Instance of LLMClient.
            generation_config (dict): Generation configuration for the language model.
            results_path (str): Path to save the results.

        Returns:
            pd.DataFrame: DataFrame containing the evaluation metrics for each strategy.
        """
        print("Evaluating One-shot Unconstrained:")
        metrics_one_shot_unconstrained = await self.evaluate_model_parallel(dataset, model, llm_client, OneShotUnconstrainedCorrection(), generation_config, results_path)

        print("Evaluating One-shot Closest:")
        metrics_one_shot_closest = await self.evaluate_model_parallel(dataset, model, llm_client, OneShotClosestCorrection(self.metrics_calculator), generation_config, results_path) # Pass metrics_calculator

        print("Evaluating Oracle:")
        metrics_oracle = await self.evaluate_model_parallel(dataset, model, llm_client, OracleHypothesisSelection(self.metrics_calculator), generation_config, results_path) # Pass metrics_calculator

        print("Evaluating Top 1:")
        metrics_top1 = await self.evaluate_model_parallel(dataset, model, llm_client, Top1HypothesisSelection(), generation_config, results_path)

        results_table = {
            "Top 1": metrics_top1,
            "One-shot Uncon": metrics_one_shot_unconstrained,
            "One-shot Closest": metrics_one_shot_closest,
            "Oracle": metrics_oracle,
        }
        results_table = pd.DataFrame.from_dict(results_table, orient='index')
        results_table = results_table[['WER', 'METEOR', 'BERT Precision', 'BERT Recall', 'BERT F1']]
        return results_table