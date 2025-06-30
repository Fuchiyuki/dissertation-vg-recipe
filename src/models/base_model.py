# src/models/base_model.py

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import openai
from dataclasses import dataclass


@dataclass
class RecipeStep:
    """Represents a single step in a recipe."""
    instruction: str
    action: str
    input_truth: Optional[str] = None
    output_truth: Optional[str] = None
    input_pred: Optional[str] = None
    output_pred: Optional[str] = None


@dataclass
class Recipe:
    """Represents a complete recipe."""
    title: str
    ingredients: List[str]
    steps: List[RecipeStep]
    recipe_id: Optional[str] = None


class BaseModel(ABC):
    """Abstract base class for recipe reasoning models."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the base model.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = 0.0  # For reproducibility
        
    @abstractmethod
    def predict_step(self, recipe: Recipe, step_index: int) -> Tuple[str, str]:
        """
        Predict input and output for a specific step.
        
        Args:
            recipe: The complete recipe
            step_index: Index of the step to predict
            
        Returns:
            Tuple of (predicted_input, predicted_output)
        """
        pass
    
    @abstractmethod
    def predict_recipe(self, recipe: Recipe) -> Recipe:
        """
        Predict inputs and outputs for all steps in a recipe.
        
        Args:
            recipe: The recipe with instructions and actions
            
        Returns:
            Recipe with predicted inputs and outputs
        """
        pass
    
    def _call_openai(self, messages: List[Dict[str, str]], 
                     max_tokens: int = 150) -> str:
        """
        Make a call to the OpenAI API.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in response
            
        Returns:
            The model's response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return ""
    
    def serialize_recipe_table(self, recipe: Recipe, 
                             mask_inputs: bool = True,
                             up_to_step: Optional[int] = None) -> str:
        """
        Convert recipe to table format as described in the paper.
        
        Args:
            recipe: The recipe to serialize
            mask_inputs: Whether to mask input/output with tokens
            up_to_step: Only include steps up to this index
            
        Returns:
            Serialized recipe string
        """
        lines = []
        max_step = len(recipe.steps) if up_to_step is None else up_to_step
        
        for i, step in enumerate(recipe.steps[:max_step]):
            if mask_inputs and (step.input_pred is None or step.output_pred is None):
                input_str = f"<in{i}>"
                output_str = f"<out{i}>"
            else:
                input_str = step.input_pred or step.input_truth or f"<in{i}>"
                output_str = step.output_pred or step.output_truth or f"<out{i}>"
            
            line = f"{step.instruction} <s> {input_str} <s> {step.action} <s> {output_str}"
            lines.append(line)
        
        return " <n> ".join(lines)
    
    def parse_prediction(self, prediction: str, step_index: int) -> Tuple[str, str]:
        """
        Parse model prediction to extract input and output.
        
        Args:
            prediction: The model's prediction string
            step_index: The step index for parsing tokens
            
        Returns:
            Tuple of (input, output)
        """
        # Look for explicit markers
        input_marker = f"<in{step_index}>:"
        output_marker = f"<out{step_index}>:"
        
        pred_input = "NA"
        pred_output = "NA"
        
        if input_marker in prediction and output_marker in prediction:
            # Extract between markers
            input_start = prediction.find(input_marker) + len(input_marker)
            input_end = prediction.find(output_marker)
            if input_end > input_start:
                pred_input = prediction[input_start:input_end].strip()
            
            output_start = prediction.find(output_marker) + len(output_marker)
            pred_output = prediction[output_start:].strip()
        else:
            # Try to parse structured format
            lines = prediction.strip().split('\n')
            for line in lines:
                if 'input:' in line.lower():
                    pred_input = line.split(':', 1)[1].strip()
                elif 'output:' in line.lower():
                    pred_output = line.split(':', 1)[1].strip()
        
        # Clean up common formatting issues
        pred_input = pred_input.strip('()[]').strip()
        pred_output = pred_output.strip('()[]').strip()
        
        return pred_input, pred_output