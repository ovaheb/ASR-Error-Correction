# Speech Recognition Post-Processing with LLaMA 3.1  

This project explores automatic speech recognition (ASR) post-processing using LLaMA 3.1. The goal is to refine ASR-generated hypotheses to improve transcription quality, evaluated using multiple metrics such as Word Error Rate (WER), METEOR, and BERTScore.  

## Features  
- **ASR Hypothesis Refinement**: Improves ASR-generated transcriptions using LLaMA 3.1.  
- **Multiple Prompting Strategies**: Includes unconstrained zero-shot and constrained zero-shot approaches.  
- **Evaluation Metrics**: Computes WER, METEOR, and BERTScore to assess post-processing effectiveness.  

## Requirements  
Install dependencies via:  

```bash
pip install -r requirements.txt
