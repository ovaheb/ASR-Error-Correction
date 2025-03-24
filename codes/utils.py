# Importing libraries
import numpy as np
import os
import inspect
import random
from time import sleep
from typing import List, Callable
import jiwer
from jiwer import wer
import re
import openai
from openai import RateLimitError
import pandas as pd
from datasets import Dataset
from sacrebleu import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer
import evaluate
from tqdm.notebook import tqdm
import logging
from transformers import AutoTokenizer
import torch
import string
import time
import asyncio
import nest_asyncio
from dotenv import load_dotenv

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Evaluation Metrics
def compute_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)


def compute_bleu(references, hypotheses):
    return corpus_bleu(hypotheses, references).score


def compute_meteor(references, hypotheses):
    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(meteor_score([ref.split()], hyp.split()))
    return sum(scores)/len(scores)


def compute_bertscore(references, hypotheses):
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    p, r, f1 = scorer.score(hypotheses, references)
    bert_score = {'precision': p.mean().item(),
                  'recall': r.mean().item(),
                  'f1': f1.mean().item()}
    return bert_score


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein distance between two strings."""
    
    len_s1, len_s2 = len(s1), len(s2)
    dp = np.zeros((len_s1 + 1, len_s2 + 1), dtype=int)

    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                           dp[i][j - 1] + 1,      # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution

    return dp[len_s1][len_s2]

# Helper functions for LLM processing
def construct_input(question):
    prompt = [{"role": "user", "content": question}]
    return prompt


def extract_hypotheses(dataset, idx):
    if 'source' in dataset.features:
        hypotheses = [h.strip() for h in dataset['source'][idx].split('.') if h.strip()]
        references = dataset['target'][idx]
    else:
        hypotheses = dataset['input'][idx]
        references = dataset['output'][idx]
        
    return hypotheses, references


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def clean_deepseek_output(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def clean_asr_output(text):
    text = re.split(r'\n+', text, maxsplit=1)[0]
    text = text.strip()
    return text

def preprocess(text):
    return clean_asr_output(remove_punctuation(text.lower()))

# Saving model results
def save_results(dataset: Dataset, corrections: list, model_name: str, function_name: str, file_path: str):
    correction_column = f"corrected_by_{model_name}_{function_name}"
    if os.path.exists(file_path):
        existing_df = pd.read_json(file_path)
    else:
        existing_df = dataset.to_pandas()

    if correction_column not in existing_df.columns:
        existing_df[correction_column] = None

    existing_df[correction_column] = corrections
    existing_df.to_json(file_path, orient="records", indent=4)
    print(f"Results saved to {file_path}")
    
# Helper functions to get model prediction
async def call_openai_with_retry(messages, model, generation_config, client):
    """Handles API retries with exponential backoff."""
    
    retry_delay = 0.1  # Initial delay in seconds
    max_delay = 10
    while True:
        try:
            # Attempt to make the API call
            generation = await client.chat.completions.create(
                model=model,
                messages=messages,
                **generation_config
            )
            return generation

        except RateLimitError as e:
            wait_time = retry_delay
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    wait_time = float(error_data.get("detail", {}).get("wait_seconds", {}))
                except:
                    pass
            await asyncio.sleep(wait_time + 0.1)


        except Exception as e:
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_delay)  # Exponential backoff


async def get_prediction(client: openai.AsyncOpenAI, model: str, messages: List[dict], generation_config: dict) -> str:
    """Asynchronously fetch predictions from OpenAI API."""
    
    try:
        generation = await call_openai_with_retry(messages, model, generation_config, client)
        return generation.choices[0].message.content if generation else ""
    except Exception as e:
        print(f"Error: {e}")
        return ""
    
    
## Implementation of out-of-sync functions with current Jiwer version
from collections import defaultdict
from typing import List, Optional, Union
from jiwer.process import AlignmentChunk, CharacterOutput, WordOutput

def visualize_error_counts(output, show_substitutions=True, show_insertions=True, show_deletions=True, top_k=None):
    s, i, d = collect_error_counts(output)

    def build_list(errors: dict):
        if len(errors) == 0:
            return "none"

        keys = [k for k in errors.keys()]
        keys = sorted(keys, reverse=True, key=lambda k: errors[k])

        if top_k is not None:
            keys = keys[:top_k]

        # we get the maximum length of all words to nicely pad output
        ln = max(len(k) if isinstance(k, str) else max(len(e) for e in k) for k in keys)

        # here we construct the string
        build = ""

        for count, (k, v) in enumerate(
            sorted(errors.items(), key=lambda tpl: tpl[1], reverse=True)
        ):
            if top_k is not None and count >= top_k:
                break

            if isinstance(k, tuple):
                build += f"{k[0]: <{ln}} --> {k[1]:<{ln}} = {v}x\n"
            else:
                build += f"{k:<{ln}} = {v}x\n"

        return build

    output = ""

    if show_substitutions:
        if output != "":
            output += "\n"
        output += "=== SUBSTITUTIONS ===\n"
        output += build_list(s)

    if show_insertions:
        if output != "":
            output += "\n"
        output += "=== INSERTIONS ===\n"
        output += build_list(i)

    if show_deletions:
        if output != "":
            output += "\n"
        output += "=== DELETIONS ===\n"
        output += build_list(d)

    if output[-1:] == "\n":
        output = output[:-1]

    return output

def collect_error_counts(output: Union[WordOutput, CharacterOutput]):
    substitutions = defaultdict(lambda: 0)
    insertions = defaultdict(lambda: 0)
    deletions = defaultdict(lambda: 0)

    for idx, sentence_chunks in enumerate(output.alignments):
        ref = output.references[idx]
        hyp = output.hypotheses[idx]
        sep = " " if isinstance(output, WordOutput) else ""

        for chunk in sentence_chunks:
            if chunk.type == "insert":
                inserted = sep.join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx])
                insertions[inserted] += 1
            if chunk.type == "delete":
                deleted = sep.join(ref[chunk.ref_start_idx : chunk.ref_end_idx])
                deletions[deleted] += 1
            if chunk.type == "substitute":
                replaced = sep.join(ref[chunk.ref_start_idx : chunk.ref_end_idx])
                by = sep.join(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx])
                substitutions[(replaced, by)] += 1

    return substitutions, insertions, deletions


async def check_availability(client, model):
    """Check if model and client is available. If model is not yet available, try again after some delay."""
    output = None
    while output is None:
        try:
            output = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Please introduce yourself."}],
            )

        except openai.APIError as e:
            print(e)
            sleep(10)

    print(output.choices[0].message.content)


if __name__ == "__main__":
    txt = "Hi, there!"
    print(remove_punctuation(txt))