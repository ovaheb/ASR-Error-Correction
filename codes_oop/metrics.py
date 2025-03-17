import jiwer
import numpy as np
from sacrebleu import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer
import nltk
from typing import List


nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class MetricsCalculator:
    """
    A class to compute various evaluation metrics for ASR error correction.
    """

    def compute_wer(self, reference: str, hypothesis: str) -> float:
        """
        Compute the Word Error Rate (WER) between a reference and a hypothesis.

        Args:
            reference (str): The reference transcript.
            hypothesis (str): The hypothesis transcript.

        Returns:
            float: The WER score.
        """
        return jiwer.wer(reference, hypothesis)
  
    def compute_bleu(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Compute the BLEU score for a list of references and hypotheses.

        Args:
            references (List[str]): A list of reference transcripts.
            hypotheses (List[str]): A list of hypothesis transcripts.

        Returns:
            float: The BLEU score.
        """
        return corpus_bleu(hypotheses, references).score

    def compute_meteor(self, references: List[str], hypotheses: List[str]) -> float:
        """
        Compute the METEOR score for a list of references and hypotheses.

        Args:
            references (List[str]): A list of reference transcripts.
            hypotheses (List[str]): A list of hypothesis transcripts.

        Returns:
            float: The average METEOR score.
        """
        scores = []
        for ref, hyp in zip(references, hypotheses):
            scores.append(meteor_score([ref.split()], hyp.split()))
        return sum(scores) / len(scores)
  
    def compute_bertscore(self, references: List[str], hypotheses: List[str]) -> dict:
        """
        Compute BERTScore for a list of references and hypotheses.

        Args:
            references (List[str]): A list of reference transcripts.
            hypotheses (List[str]): A list of hypothesis transcripts.

        Returns:
            dict: A dictionary containing precision, recall, and F1 BERTScore values.
        """
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        p, r, f1 = scorer.score(hypotheses, references)
        bert_score = {'precision': p.mean().item(),
                      'recall': r.mean().item(),
                      'f1': f1.mean().item()}
        return bert_score

    def compute_levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Compute the Levenshtein distance between two strings.

        Args:
            s1 (str): The first string.
            s2 (str): The second string.

        Returns:
            int: The Levenshtein distance.
        """
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