# src/models/cot_model.py

from typing import Tuple, List, Dict
from .base_model import BaseModel, Recipe, RecipeStep


class ChainOfThoughtModel(BaseModel):
    """Chain-of-Thought implementation for PizzaCommonsense."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model_name)
        
    def predict_step(self, recipe: Recipe, step_index: int) -> Tuple[str, str]:
        """
        Predict input and output for a specific step using CoT reasoning.
        
        Args:
            recipe: The complete recipe
            step_index: Index of the step to predict
            
        Returns:
            Tuple of (predicted_input, predicted_output)
        """
        # Build context from previous steps
        context = self._build_context(recipe, step_index)
        
        # Create CoT prompt
        prompt = self._create_cot_prompt(recipe.steps[step_index], context)
        
        # Get prediction
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        prediction = self._call_openai(messages, max_tokens=300)
        
        # Parse prediction
        pred_input, pred_output = self.parse_prediction(prediction, step_index)
        
        return pred_input, pred_output
    
    def predict_recipe(self, recipe: Recipe) -> Recipe:
        """
        Predict inputs and outputs for all steps in a recipe using CoT.
        
        Args:
            recipe: The recipe with instructions and actions
            
        Returns:
            Recipe with predicted inputs and outputs
        """
        # Process each step sequentially
        for i in range(len(recipe.steps)):
            pred_input, pred_output = self.predict_step(recipe, i)
            recipe.steps[i].input_pred = pred_input
            recipe.steps[i].output_pred = pred_output
            
        return recipe
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for CoT reasoning."""
        return """You are an expert cooking assistant that understands recipe instructions and can reason about the inputs and outputs of each cooking step.

For each cooking step, you need to:
1. Identify what ingredients or preparations are needed as input
2. Determine what the result/output of the action will be

Rules:
- Input and output should only contain comestibles (food items), not tools or locations
- Use "NA" for steps that don't involve food transformation (e.g., "preheat oven")
- Be specific about the state of ingredients (e.g., "diced onions" not just "onions")
- Consider outputs from previous steps that might be inputs to the current step
- Use semicolons to separate multiple items (e.g., "flour; water; yeast")"""
    
    def _build_context(self, recipe: Recipe, up_to_step: int) -> List[Dict[str, str]]:
        """Build context from previous steps."""
        context = []
        
        # Add ingredients
        context.append({
            "type": "ingredients",
            "content": ", ".join(recipe.ingredients) if recipe.ingredients else "Not specified"
        })
        
        # Add previous steps with their predictions
        for i in range(up_to_step):
            step = recipe.steps[i]
            context.append({
                "type": "step",
                "index": i,
                "instruction": step.instruction,
                "action": step.action,
                "input": step.input_pred or "Unknown",
                "output": step.output_pred or "Unknown"
            })
        
        return context
    
    def _create_cot_prompt(self, step: RecipeStep, context: List[Dict[str, str]]) -> str:
        """Create a Chain-of-Thought prompt for the step."""
        prompt_parts = []
        
        # Add context
        prompt_parts.append("Let's think step by step about this cooking recipe.\n")
        
        # Add ingredients
        ingredients_info = next((c for c in context if c["type"] == "ingredients"), None)
        if ingredients_info:
            prompt_parts.append(f"Available ingredients: {ingredients_info['content']}\n")
        
        # Add previous steps
        prev_steps = [c for c in context if c["type"] == "step"]
        if prev_steps:
            prompt_parts.append("\nPrevious steps completed:")
            for ctx_step in prev_steps:
                prompt_parts.append(f"Step {ctx_step['index'] + 1}: {ctx_step['instruction']}")
                prompt_parts.append(f"  - Input: {ctx_step['input']}")
                prompt_parts.append(f"  - Output: {ctx_step['output']}")
        
        # Add current step
        prompt_parts.append(f"\nCurrent step: {step.instruction}")
        prompt_parts.append(f"Cooking action: {step.action}")
        
        # Add reasoning request
        prompt_parts.append("\nLet's reason about this step:")
        prompt_parts.append("1. What ingredients or preparations from previous steps are needed as input?")
        prompt_parts.append("2. What will be the result after performing this action?")
        prompt_parts.append("\nThink carefully about:")
        prompt_parts.append("- Implicit ingredients (e.g., 'season' implies salt/pepper)")
        prompt_parts.append("- Outputs from previous steps that might be used here")
        prompt_parts.append("- The transformation caused by the cooking action")
        
        prompt_parts.append(f"\nProvide your answer in this format:")
        prompt_parts.append(f"<in{len(prev_steps)}>: [your input prediction]")
        prompt_parts.append(f"<out{len(prev_steps)}>: [your output prediction]")
        
        return "\n".join(prompt_parts)
    
    def predict_with_self_consistency(self, recipe: Recipe, step_index: int, 
                                    num_samples: int = 5) -> Tuple[str, str]:
        """
        Use self-consistency by sampling multiple predictions.
        
        Args:
            recipe: The complete recipe
            step_index: Index of the step to predict
            num_samples: Number of samples to generate
            
        Returns:
            Most consistent prediction
        """
        predictions = []
        
        # Save original temperature
        original_temp = self.temperature
        self.temperature = 0.7  # Increase for sampling
        
        for _ in range(num_samples):
            pred_input, pred_output = self.predict_step(recipe, step_index)
            predictions.append((pred_input, pred_output))
        
        # Restore temperature
        self.temperature = original_temp
        
        # Find most common prediction (simple majority vote)
        from collections import Counter
        input_counts = Counter([p[0] for p in predictions])
        output_counts = Counter([p[1] for p in predictions])
        
        most_common_input = input_counts.most_common(1)[0][0]
        most_common_output = output_counts.most_common(1)[0][0]
        
        return most_common_input, most_common_output