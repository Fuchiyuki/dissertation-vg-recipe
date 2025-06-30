# experiments/run_experiments.py

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.cot_model import ChainOfThoughtModel
from src.models.cok_model import ChainOfKnowledgeModel
from src.models.base_model import Recipe, RecipeStep
from src.evaluation.metrics import PizzaCommonsenseMetrics
import config


class PizzaCommonsenseExperiment:
    """Main experiment runner for comparing CoT and CoK on PizzaCommonsense."""
    
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = PizzaCommonsenseMetrics()
        self.api_key = config.OPENAI_API_KEY
        
    def load_data(self, split: str = 'test') -> List[Recipe]:
        """Load PizzaCommonsense data."""
        # This is a placeholder - adapt based on actual data format
        file_path = self.data_path / f'{split}.json'
        
        recipes = []
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for recipe_data in data:
            steps = []
            for step_data in recipe_data['steps']:
                step = RecipeStep(
                    instruction=step_data['instruction'],
                    action=step_data['action'],
                    input_truth=step_data.get('input', 'NA'),
                    output_truth=step_data.get('output', 'NA')
                )
                steps.append(step)
            
            recipe = Recipe(
                title=recipe_data.get('title', 'Unknown Recipe'),
                ingredients=recipe_data.get('ingredients', []),
                steps=steps,
                recipe_id=recipe_data.get('id')
            )
            recipes.append(recipe)
        
        return recipes
    
    def run_cot_experiment(self, recipes: List[Recipe], 
                          model_name: str = "gpt-3.5-turbo") -> Dict:
        """Run Chain-of-Thought experiment."""
        print(f"\nRunning CoT experiment with {model_name}...")
        
        model = ChainOfThoughtModel(self.api_key, model_name)
        results = {
            'method': 'CoT',
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'predictions': [],
            'metrics': {}
        }
        
        all_predictions = []
        all_references = []
        
        for recipe in tqdm(recipes, desc="Processing recipes"):
            # Get predictions
            predicted_recipe = model.predict_recipe(recipe)
            
            # Collect predictions and references
            recipe_result = {
                'recipe_id': recipe.recipe_id,
                'title': recipe.title,
                'steps': []
            }
            
            for step in predicted_recipe.steps:
                all_predictions.append((step.input_pred, step.output_pred))
                all_references.append((step.input_truth, step.output_truth))
                
                recipe_result['steps'].append({
                    'instruction': step.instruction,
                    'action': step.action,
                    'input_truth': step.input_truth,
                    'output_truth': step.output_truth,
                    'input_pred': step.input_pred,
                    'output_pred': step.output_pred
                })
            
            results['predictions'].append(recipe_result)
        
        # Compute metrics
        results['metrics'] = self.metrics.compute_all_metrics(
            all_predictions, all_references
        )
        
        # Save results
        output_file = self.output_dir / f'cot_{model_name}_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_cok_experiment(self, recipes: List[Recipe], 
                          model_name: str = "gpt-3.5-turbo") -> Dict:
        """Run Chain-of-Knowledge experiment."""
        print(f"\nRunning CoK experiment with {model_name}...")
        
        model = ChainOfKnowledgeModel(self.api_key, model_name)
        results = {
            'method': 'CoK',
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'predictions': [],
            'metrics': {}
        }
        
        all_predictions = []
        all_references = []
        
        for recipe in tqdm(recipes, desc="Processing recipes"):
            # Get predictions
            predicted_recipe = model.predict_recipe(recipe)
            
            # Collect predictions and references
            recipe_result = {
                'recipe_id': recipe.recipe_id,
                'title': recipe.title,
                'steps': []
            }
            
            for step in predicted_recipe.steps:
                all_predictions.append((step.input_pred, step.output_pred))
                all_references.append((step.input_truth, step.output_truth))
                
                recipe_result['steps'].append({
                    'instruction': step.instruction,
                    'action': step.action,
                    'input_truth': step.input_truth,
                    'output_truth': step.output_truth,
                    'input_pred': step.input_pred,
                    'output_pred': step.output_pred
                })
            
            results['predictions'].append(recipe_result)
        
        # Compute metrics
        results['metrics'] = self.metrics.compute_all_metrics(
            all_predictions, all_references
        )
        
        # Save results
        output_file = self.output_dir / f'cok_{model_name}_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def compare_results(self, cot_results: Dict, cok_results: Dict) -> pd.DataFrame:
        """Compare CoT and CoK results."""
        comparison = []
        
        # Extract metrics
        for metric_name in cot_results['metrics'].keys():
            comparison.append({
                'Metric': metric_name,
                'CoT': f"{cot_results['metrics'][metric_name]:.2f}",
                'CoK': f"{cok_results['metrics'][metric_name]:.2f}",
                'Improvement': f"{cok_results['metrics'][metric_name] - cot_results['metrics'][metric_name]:.2f}"
            })
        
        df = pd.DataFrame(comparison)
        
        # Save comparison
        output_file = self.output_dir / f'comparison_{datetime.now():%Y%m%d_%H%M%S}.csv'
        df.to_csv(output_file, index=False)
        
        return df
    
    def run_full_experiment(self, num_recipes: int = None, model_name: str = "gpt-3.5-turbo"):
        """Run full experiment comparing CoT and CoK."""
        print("Starting PizzaCommonsense experiment...")
        
        # Load data
        recipes = self.load_data('test')
        if num_recipes:
            recipes = recipes[:num_recipes]
        print(f"Loaded {len(recipes)} recipes")
        
        # Run CoT
        cot_results = self.run_cot_experiment(recipes, model_name)
        print("\nCoT Results:")
        for metric, value in cot_results['metrics'].items():
            print(f"  {metric}: {value:.2f}")
        
        # Run CoK
        cok_results = self.run_cok_experiment(recipes, model_name)
        print("\nCoK Results:")
        for metric, value in cok_results['metrics'].items():
            print(f"  {metric}: {value:.2f}")
        
        # Compare
        comparison_df = self.compare_results(cot_results, cok_results)
        print("\nComparison:")
        print(comparison_df.to_string(index=False))
        
        return cot_results, cok_results, comparison_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run PizzaCommonsense experiments')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to PizzaCommonsense data')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--num-recipes', type=int, default=None,
                        help='Number of recipes to process (for testing)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                        choices=['gpt-3.5-turbo', 'gpt-4'],
                        help='OpenAI model to use')
    parser.add_argument('--method', type=str, default='both',
                        choices=['cot', 'cok', 'both'],
                        help='Which method to run')
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = PizzaCommonsenseExperiment(args.data_path, args.output_dir)
    
    if args.method == 'both':
        experiment.run_full_experiment(args.num_recipes, args.model)
    else:
        recipes = experiment.load_data('test')
        if args.num_recipes:
            recipes = recipes[:args.num_recipes]
        
        if args.method == 'cot':
            results = experiment.run_cot_experiment(recipes, args.model)
        else:
            results = experiment.run_cok_experiment(recipes, args.model)
        
        print(f"\n{args.method.upper()} Results:")
        for metric, value in results['metrics'].items():
            print(f"  {metric}: {value:.2f}")


if __name__ == '__main__':
    main()