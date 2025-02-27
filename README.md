# Speech Recognition Post-Processing with LLaMA 3.1

This project explores automatic speech recognition (ASR) generative error correction using LLMs. The goal is to refine ASR-generated hypotheses to improve transcription quality, evaluated using multiple metrics such as Word Error Rate (WER), METEOR, and BERTScore.

## Features
- **ASR Hypothesis Refinement**: Improves ASR-generated transcriptions using Llama 3.1 8B, Gemma 2 9B, and Mistral 7B.
- **Multiple Prompting Strategies**: Includes unconstrained zero-shot, constrained zero-shot, few-shot, chain of thought, and task-activating prompting approaches.
- **Evaluation Metrics**: Computes WER, METEOR, and BERTScore to assess post-processing effectiveness.
- **Datasets**: HP test sets.

## Requirements
Install dependencies via:

```bash
pip install -r requirements.txt
