# src/evaluation/metrics.py

from typing import List, Tuple, Dict
import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass


class PizzaCommonsenseMetrics:
    """Evaluation metrics for PizzaCommonsense task."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        
    def compute_all_metrics(self, predictions: List[Tuple[str, str]], 
                          references: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Compute all metrics for the predictions.
        
        Args:
            predictions: List of (input_pred, output_pred) tuples
            references: List of (input_truth, output_truth) tuples
            
        Returns:
            Dictionary of metric scores
        """
        # Separate inputs and outputs
        pred_inputs = [p[0] for p in predictions]
        pred_outputs = [p[1] for p in predictions]
        ref_inputs = [r[0] for r in references]
        ref_outputs = [r[1] for r in references]
        
        metrics = {}
        
        # Input metrics
        metrics['input_ema'] = self.exact_match_accuracy(pred_inputs, ref_inputs)
        metrics['input_rouge_l'] = self.rouge_l_score(pred_inputs, ref_inputs)
        
        # Output metrics
        metrics['output_ema'] = self.exact_match_accuracy(pred_outputs, ref_outputs)
        metrics['output_rouge_l'] = self.rouge_l_score(pred_outputs, ref_outputs)
        metrics['output_bleu'] = self.bleu_score(pred_outputs, ref_outputs)
        metrics['output_meteor'] = self.meteor_score(pred_outputs, ref_outputs)
        
        # BERTScore for outputs
        bert_p, bert_r, bert_f1 = self.bert_score(pred_outputs, ref_outputs)
        metrics['output_bertscore_precision'] = bert_p
        metrics['output_bertscore_recall'] = bert_r
        metrics['output_bertscore_f1'] = bert_f1
        
        return metrics
    
    def exact_match_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute exact match accuracy.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Exact match accuracy (0-100)
        """
        if not predictions or not references:
            return 0.0
        
        matches = sum(1 for p, r in zip(predictions, references) 
                     if self._normalize_string(p) == self._normalize_string(r))
        
        return (matches / len(predictions)) * 100
    
    def rouge_l_score(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute ROUGE-L score.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Average ROUGE-L F1 score (0-100)
        """
        if not predictions or not references:
            return 0.0
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            scores.append(score['rougeL'].fmeasure)
        
        return np.mean(scores) * 100
    
    def bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            BLEU score (0-100)
        """
        if not predictions or not references:
            return 0.0
        
        # BLEU expects references to be a list of lists
        refs = [[ref] for ref in references]
        
        try:
            score = self.bleu.corpus_score(predictions, refs)
            return score.score
        except:
            return 0.0
    
    def meteor_score(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute METEOR score.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Average METEOR score (0-100)
        """
        if not predictions or not references:
            return 0.0
        
        scores = []
        for pred, ref in zip(predictions, references):
            try:
                # Tokenize
                pred_tokens = word_tokenize(pred.lower())
                ref_tokens = word_tokenize(ref.lower())
                
                # Compute METEOR
                score = meteor_score([ref_tokens], pred_tokens)
                scores.append(score)
            except:
                scores.append(0.0)
        
        return np.mean(scores) * 100
    
    def bert_score(self, predictions: List[str], references: List[str]) -> Tuple[float, float, float]:
        """
        Compute BERTScore.
        
        Args:
            predictions: List of predicted strings
            references: List of reference strings
            
        Returns:
            Tuple of (precision, recall, f1) scores (0-100)
        """
        if not predictions or not references:
            return 0.0, 0.0, 0.0
        
        try:
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
            return (
                P.mean().item() * 100,
                R.mean().item() * 100,
                F1.mean().item() * 100
            )
        except:
            return 0.0, 0.0, 0.0
    
    def _normalize_string(self, s: str) -> str:
        """Normalize string for comparison."""
        # Convert to lowercase and strip whitespace
        s = s.lower().strip()
        
        # Remove extra spaces
        s = ' '.join(s.split())
        
        # Remove parentheses and brackets
        s = s.replace('(', '').replace(')', '')
        s = s.replace('[', '').replace(']', '')
        
        # Normalize 'NA' variations
        if s in ['na', 'n/a', 'none', 'n.a.']:
            s = 'na'
        
        return s
    
    def compute_step_level_metrics(self, prediction: Tuple[str, str], 
                                 reference: Tuple[str, str]) -> Dict[str, float]:
        """
        Compute metrics for a single step prediction.
        
        Args:
            prediction: (input_pred, output_pred) tuple
            reference: (input_truth, output_truth) tuple
            
        Returns:
            Dictionary of metric scores for this step
        """
        pred_input, pred_output = prediction
        ref_input, ref_output = reference
        
        metrics = {}
        
        # Input metrics
        metrics['input_exact_match'] = float(
            self._normalize_string(pred_input) == self._normalize_string(ref_input)
        )
        metrics['input_rouge_l'] = self.rouge_scorer.score(
            ref_input, pred_input
        )['rougeL'].fmeasure
        
        # Output metrics
        metrics['output_exact_match'] = float(
            self._normalize_string(pred_output) == self._normalize_string(ref_output)
        )
        metrics['output_rouge_l'] = self.rouge_scorer.score(
            ref_output, pred_output
        )['rougeL'].fmeasure
        
        return metrics