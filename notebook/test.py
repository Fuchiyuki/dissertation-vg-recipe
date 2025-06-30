# notebooks/example_usage.py
# Convert to .ipynb or run as script

"""
Example usage of the Chain-of-Knowledge system for PizzaCommonsense
"""

import sys
from pathlib import Path
sys.path.append(str(Path('.').parent))

from src.models.cot_model import ChainOfThoughtModel
from src.models.cok_model import ChainOfKnowledgeModel
from src.models.base_model import Recipe, RecipeStep
from src.evaluation.metrics import PizzaCommonsenseMetrics
import config

# %% [markdown]
# # Example: Using Chain-of-Knowledge for PizzaCommonsense
# This notebook demonstrates how to use both CoT and CoK models on a sample recipe.

# %%
# Create a sample recipe
sample_recipe = Recipe(
    title="Simple Margherita Pizza",
    ingredients=[
        "2 cups flour",
        "1 cup warm water", 
        "1 packet yeast",
        "1 tsp salt",
        "2 tbsp olive oil",
        "1 cup tomato sauce",
        "8 oz mozzarella cheese",
        "fresh basil leaves"
    ],
    steps=[
        RecipeStep(
            instruction="Mix warm water and yeast in a bowl",
            action="mix",
            input_truth="warm water; yeast",
            output_truth="yeast mixture"
        ),
        RecipeStep(
            instruction="Combine flour and salt in a large bowl",
            action="combine", 
            input_truth="flour; salt",
            output_truth="flour and salt mixture"
        ),
        RecipeStep(
            instruction="Add the yeast mixture to the flour",
            action="add",
            input_truth="yeast mixture; flour and salt mixture",
            output_truth="wet dough mixture"
        ),
        RecipeStep(
            instruction="Knead the dough for 10 minutes",
            action="knead",
            input_truth="wet dough mixture",
            output_truth="smooth elastic dough"
        ),
        RecipeStep(
            instruction="Spread tomato sauce on the dough",
            action="spread",
            input_truth="tomato sauce; smooth elastic dough",
            output_truth="sauced pizza dough"
        ),
        RecipeStep(
            instruction="Sprinkle mozzarella cheese on top",
            action="sprinkle",
            input_truth="mozzarella cheese; sauced pizza dough",
            output_truth="assembled pizza"
        ),
        RecipeStep(
            instruction="Bake in the oven at 450F for 12-15 minutes",
            action="bake",
            input_truth="assembled pizza",
            output_truth="baked margherita pizza"
        )
    ]
)

# %%
# Test Chain-of-Thought model
print("Testing Chain-of-Thought Model")
print("=" * 50)

cot_model = ChainOfThoughtModel(config.OPENAI_API_KEY)

# Predict for a single step
step_index = 2  # "Add the yeast mixture to the flour"
print(f"\nPredicting step {step_index + 1}: {sample_recipe.steps[step_index].instruction}")

pred_input, pred_output = cot_model.predict_step(sample_recipe, step_index)
print(f"Predicted input: {pred_input}")
print(f"Actual input: {sample_recipe.steps[step_index].input_truth}")
print(f"Predicted output: {pred_output}")
print(f"Actual output: {sample_recipe.steps[step_index].output_truth}")

# %%
# Test Chain-of-Knowledge model
print("\n\nTesting Chain-of-Knowledge Model")
print("=" * 50)

cok_model = ChainOfKnowledgeModel(config.OPENAI_API_KEY)

# Predict for the same step
print(f"\nPredicting step {step_index + 1}: {sample_recipe.steps[step_index].instruction}")

pred_input, pred_output = cok_model.predict_step(sample_recipe, step_index)
print(f"Predicted input: {pred_input}")
print(f"Actual input: {sample_recipe.steps[step_index].input_truth}")
print(f"Predicted output: {pred_output}")
print(f"Actual output: {sample_recipe.steps[step_index].output_truth}")

# %%
# Compare full recipe predictions
print("\n\nComparing Full Recipe Predictions")
print("=" * 50)

# Reset predictions
for step in sample_recipe.steps:
    step.input_pred = None
    step.output_pred = None

# CoT predictions
cot_recipe = cot_model.predict_recipe(sample_recipe)
print("\nCoT Predictions:")
for i, step in enumerate(cot_recipe.steps):
    print(f"\nStep {i+1}: {step.instruction}")
    print(f"  Predicted: {step.input_pred} -> {step.output_pred}")
    print(f"  Actual:    {step.input_truth} -> {step.output_truth}")

# Reset for CoK
for step in sample_recipe.steps:
    step.input_pred = None
    step.output_pred = None

# CoK predictions
cok_recipe = cok_model.predict_recipe(sample_recipe)
print("\n\nCoK Predictions:")
for i, step in enumerate(cok_recipe.steps):
    print(f"\nStep {i+1}: {step.instruction}")
    print(f"  Predicted: {step.input_pred} -> {step.output_pred}")
    print(f"  Actual:    {step.input_truth} -> {step.output_truth}")

# %%
# Evaluate metrics
print("\n\nEvaluation Metrics")
print("=" * 50)

metrics = PizzaCommonsenseMetrics()

# Collect predictions and references
cot_predictions = [(s.input_pred, s.output_pred) for s in cot_recipe.steps]
cok_predictions = [(s.input_pred, s.output_pred) for s in cok_recipe.steps]
references = [(s.input_truth, s.output_truth) for s in sample_recipe.steps]

# Compute metrics
cot_metrics = metrics.compute_all_metrics(cot_predictions, references)
cok_metrics = metrics.compute_all_metrics(cok_predictions, references)

print("\nCoT Metrics:")
for metric, value in cot_metrics.items():
    print(f"  {metric}: {value:.2f}")

print("\nCoK Metrics:")
for metric, value in cok_metrics.items():
    print(f"  {metric}: {value:.2f}")

print("\nImprovement (CoK - CoT):")
for metric in cot_metrics:
    improvement = cok_metrics[metric] - cot_metrics[metric]
    print(f"  {metric}: {improvement:+.2f}")

# %%
# Demonstrate knowledge retrieval
print("\n\nKnowledge Retrieval Example")
print("=" * 50)

from src.knowledge.knowledge_sources import CookingKnowledgeBase
from src.knowledge.query_generator import QueryGenerator

kb = CookingKnowledgeBase()
qg = QueryGenerator(config.OPENAI_API_KEY)

# Generate queries for a step
step = sample_recipe.steps[2]  # "Add the yeast mixture to the flour"
queries = qg.generate_queries(step, sample_recipe)

print(f"Step: {step.instruction}")
print(f"\nGenerated queries:")
for i, query in enumerate(queries):
    print(f"  {i+1}. {query}")

print(f"\nRetrieved knowledge:")
for query in queries[:2]:  # Show first 2 queries
    knowledge = kb.retrieve(query)
    print(f"\nQuery: {query}")
    for k in knowledge[:2]:  # Show top 2 results
        print(f"  - {k['content'][:100]}...")

# %%
print("\n\nExample complete! This demonstrates the basic usage of CoT and CoK models.")