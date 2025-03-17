import os
import pandas as pd
import asyncio
import nest_asyncio
from datasets import Dataset
from dotenv import load_dotenv
from time import sleep
import logging

from metrics import MetricsCalculator
from data_handler import DataHandler
from llm_client import LLMClient
from correction_strategies import (
    ZeroShotUnconstrainedCorrection,
    ZeroShotConstrainedCorrection,
    ZeroShotClosestCorrection,
    OracleHypothesisSelection,
    Top1HypothesisSelection
)
from evaluation_pipeline import EvaluationPipeline
from utils import ProgressTracker


load_dotenv()
nest_asyncio.apply()
logging.getLogger("transformers").setLevel(logging.ERROR)


async def main():
    """
    Main function to run the ASR error correction evaluation pipeline.
    """

    # Initialize components
    metrics_calculator = MetricsCalculator()
    data_handler = DataHandler()
    progress_tracker = ProgressTracker()
    llm_client = LLMClient(api_key=os.environ.get("OPENAI_API_KEY"))

    evaluation_pipeline = EvaluationPipeline(metrics_calculator, data_handler, progress_tracker)


    model = "Meta-Llama-3.1-8B-Instruct" # Or "gpt-3.5-turbo" if you want to use OpenAI models.
    small_generation_config = {"max_tokens": 20, "temperature": 0.9}
    #moderate_generation_config = {"max_tokens": 200, "temperature": 0.9} # If needed later

    # Basic model availability check (you can remove or adjust this)
    output = None
    while output is None:
        try:
            output = await llm_client.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Please introduce yourself."}],
            )
        except openai.APIError as e: # Assuming openai is imported in main if you are using openai check.
            print(e)
            sleep(10)
    print(output.choices[0].message.content)

    # Load Dataset - adjust path if needed
    df = pd.read_csv("~/projects/ASR-Error-Correction/data/test_cv.csv").iloc[:100]
    results_path = "~/projects/ASR-Error-Correction/results/test_cv.json"
    os.makedirs(os.path.expanduser("~/projects/ASR-Error-Correction/results"), exist_ok=True)
    dataset = Dataset.from_pandas(df)
    print(dataset)

    # Run Evaluation
    results_table = await evaluation_pipeline.run_evaluation(dataset, model, llm_client, small_generation_config, results_path)
    print(results_table)


if __name__ == "__main__":
    asyncio.run(main())