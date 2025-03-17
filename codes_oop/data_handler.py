import os
import pandas as pd
from datasets import Dataset
from typing import List

class DataHandler:
    """
    A class to handle data loading, processing, and saving for ASR error correction.
    """

    def __init__(self):
        pass # You can add initialization if needed later

    def extract_hypotheses(self, dataset: Dataset, idx: int) -> tuple[List[str], str]:
        """
        Extract hypotheses and references from a dataset at a given index.

        Args:
            dataset (Dataset): The dataset object.
            idx (int): The index of the data sample.

        Returns:
            tuple[List[str], str]: A tuple containing the list of hypotheses and the reference transcript.
        """
        if 'source' in dataset.features:
            hypotheses = [h.strip() for h in dataset['source'][idx].split('.') if h.strip()]
            references = dataset['target'][idx]
        else:
            hypotheses = dataset['input'][idx]
            references = dataset['output'][idx]
        return hypotheses, references

    def save_results(self, dataset: Dataset, corrections: list, model_name: str, function_name: str, file_path: str):
        """
        Save the correction results to a JSON file, merging with existing data if the file exists.

        Args:
            dataset (Dataset): The original dataset.
            corrections (list): A list of corrected transcripts.
            model_name (str): The name of the model used for correction.
            function_name (str): The name of the correction function.
            file_path (str): The path to save the results JSON file.
        """
        correction_column = f"corrected_by_{model_name}_{function_name}"

        if os.path.exists(file_path):
            existing_df = pd.read_json(file_path)
        else:
            existing_df = dataset.to_pandas()

        if correction_column not in existing_df.columns:
            existing_df[correction_column] = None

        # Update only missing values (keep previous results)
        for idx, correction in enumerate(corrections):
            if pd.isna(existing_df.at[idx, correction_column]):
                existing_df.at[idx, correction_column] = correction

        existing_df.to_json(file_path, orient="records", indent=4)
        print(f"Results saved to {file_path}")