# src/models/cok_model.py

from typing import Tuple, List, Dict, Optional
from .base_model import BaseModel, Recipe, RecipeStep
from ..knowledge.knowledge_sources import CookingKnowledgeBase
from ..knowledge.query_generator import QueryGenerator
from collections import Counter


class ChainOfKnowledgeModel(BaseModel):
    """Chain-of-Knowledge implementation for PizzaCommonsense."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model_name)
        self.knowledge_base = CookingKnowledgeBase()
        self.query_generator = QueryGenerator(api_key)
        self.consistency_threshold = 0.6  # Threshold for self-consistency
        
    def predict_recipe(self, recipe: Recipe) -> Recipe:
        """
        Predict inputs and outputs for all steps using Chain-of-Knowledge.
        
        This follows the three-stage process from the paper:
        1. Reasoning Preparation
        2. Dynamic Knowledge Adapting
        3. Answer Consolidation
        
        Args:
            recipe: The recipe with instructions and actions
            
        Returns:
            Recipe with predicted inputs and outputs
        """
        # Stage 1: Reasoning Preparation
        preliminary_predictions = self._reasoning_preparation(recipe)
        
        # Stage 2: Dynamic Knowledge Adapting
        corrected_predictions = self._dynamic_knowledge_adapting(
            recipe, preliminary_predictions
        )
        
        # Stage 3: Answer Consolidation
        final_recipe = self._answer_consolidation(recipe, corrected_predictions)
        
        return final_recipe
    
    def predict_step(self, recipe: Recipe, step_index: int) -> Tuple[str, str]:
        """
        Predict input and output for a specific step using CoK.
        
        Args:
            recipe: The complete recipe
            step_index: Index of the step to predict
            
        Returns:
            Tuple of (predicted_input, predicted_output)
        """
        # For single step prediction, we still use the full CoK process
        # but only for the steps up to and including the target step
        partial_recipe = Recipe(
            title=recipe.title,
            ingredients=recipe.ingredients,
            steps=recipe.steps[:step_index + 1]
        )
        
        result = self.predict_recipe(partial_recipe)
        step = result.steps[step_index]
        
        return step.input_pred, step.output_pred
    
    def _reasoning_preparation(self, recipe: Recipe) -> Dict[int, List[Tuple[str, str]]]:
        """
        Stage 1: Generate preliminary rationales and identify knowledge domains.
        
        Returns:
            Dictionary mapping step indices to list of (input, output) predictions
        """
        predictions = {}
        
        for i, step in enumerate(recipe.steps):
            # Generate multiple predictions for self-consistency
            step_predictions = []
            
            # Identify relevant knowledge domains
            domains = self._identify_knowledge_domains(step)
            
            # Generate preliminary predictions
            for _ in range(5):  # 5 samples for self-consistency
                pred = self._generate_preliminary_prediction(recipe, i, domains)
                step_predictions.append(pred)
            
            predictions[i] = step_predictions
        
        return predictions
    
    def _dynamic_knowledge_adapting(self, recipe: Recipe, 
                                  preliminary_predictions: Dict[int, List[Tuple[str, str]]]) -> Dict[int, Tuple[str, str]]:
        """
        Stage 2: Progressively correct rationales using retrieved knowledge.
        
        Returns:
            Dictionary mapping step indices to corrected (input, output) predictions
        """
        corrected_predictions = {}
        
        for i, step in enumerate(recipe.steps):
            # Check self-consistency
            predictions = preliminary_predictions[i]
            needs_correction = self._check_consistency(predictions) < self.consistency_threshold
            
            if needs_correction:
                # Generate queries for knowledge retrieval
                queries = self.query_generator.generate_queries(step, recipe)
                
                # Retrieve relevant knowledge
                knowledge = []
                for query in queries:
                    retrieved = self.knowledge_base.retrieve(query)
                    knowledge.extend(retrieved)
                
                # Correct the prediction using retrieved knowledge
                corrected = self._correct_with_knowledge(
                    recipe, i, predictions[0], knowledge, corrected_predictions
                )
                corrected_predictions[i] = corrected
            else:
                # Use most common prediction
                corrected_predictions[i] = self._get_consensus_prediction(predictions)
        
        return corrected_predictions
    
    def _answer_consolidation(self, recipe: Recipe, 
                            corrected_predictions: Dict[int, Tuple[str, str]]) -> Recipe:
        """
        Stage 3: Generate final answers based on corrected rationales.
        
        Returns:
            Recipe with final predictions
        """
        for i, (pred_input, pred_output) in corrected_predictions.items():
            recipe.steps[i].input_pred = pred_input
            recipe.steps[i].output_pred = pred_output
        
        return recipe
    
    def _identify_knowledge_domains(self, step: RecipeStep) -> List[str]:
        """Identify relevant knowledge domains for the step."""
        domains = []
        
        # Simple heuristic-based domain identification
        action_lower = step.action.lower()
        instruction_lower = step.instruction.lower()
        
        # Ingredient transformations
        if any(word in action_lower for word in ['mix', 'combine', 'blend', 'stir']):
            domains.append('ingredient_combination')
        
        # Cooking techniques
        if any(word in action_lower for word in ['bake', 'fry', 'boil', 'cook', 'heat']):
            domains.append('cooking_techniques')
        
        # Physical transformations
        if any(word in action_lower for word in ['chop', 'dice', 'slice', 'cut', 'grate']):
            domains.append('physical_transformation')
        
        # Temperature-related
        if any(word in instruction_lower for word in ['preheat', 'temperature', 'degrees']):
            domains.append('temperature')
        
        # Default domain
        if not domains:
            domains.append('general_cooking')
        
        return domains
    
    def _generate_preliminary_prediction(self, recipe: Recipe, step_index: int, 
                                       domains: List[str]) -> Tuple[str, str]:
        """Generate a preliminary prediction for a step."""
        context = self._build_context(recipe, step_index)
        
        prompt = f"""Given this cooking recipe step, predict the input ingredients and output result.

Recipe: {recipe.title}
Ingredients: {', '.join(recipe.ingredients)}

Previous steps:
{self._format_previous_steps(recipe, step_index)}

Current step: {recipe.steps[step_index].instruction}
Action: {recipe.steps[step_index].action}
Relevant domains: {', '.join(domains)}

Think step by step:
1. What ingredients/preparations are needed as input?
2. What is the result after this action?

Format your answer as:
Input: [ingredients/preparations needed]
Output: [result after the action]"""
        
        messages = [
            {"role": "system", "content": self._get_cok_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_openai(messages, max_tokens=200)
        return self._parse_cok_prediction(response)
    
    def _correct_with_knowledge(self, recipe: Recipe, step_index: int,
                              initial_prediction: Tuple[str, str],
                              knowledge: List[Dict[str, str]],
                              previous_corrections: Dict[int, Tuple[str, str]]) -> Tuple[str, str]:
        """Correct a prediction using retrieved knowledge."""
        # Use previous corrections for context
        context = []
        for i in range(step_index):
            if i in previous_corrections:
                recipe.steps[i].input_pred = previous_corrections[i][0]
                recipe.steps[i].output_pred = previous_corrections[i][1]
        
        # Format knowledge for prompt
        knowledge_text = self._format_knowledge(knowledge)
        
        prompt = f"""Correct this cooking step prediction using the provided knowledge.

Step: {recipe.steps[step_index].instruction}
Action: {recipe.steps[step_index].action}

Initial prediction:
Input: {initial_prediction[0]}
Output: {initial_prediction[1]}

Relevant cooking knowledge:
{knowledge_text}

Previous steps context:
{self._format_previous_steps(recipe, step_index)}

Provide a corrected prediction based on the knowledge:
Input: [corrected input]
Output: [corrected output]"""
        
        messages = [
            {"role": "system", "content": self._get_cok_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_openai(messages, max_tokens=200)
        return self._parse_cok_prediction(response)
    
    def _check_consistency(self, predictions: List[Tuple[str, str]]) -> float:
        """Check consistency among multiple predictions."""
        if len(predictions) <= 1:
            return 1.0
        
        # Count occurrences of each prediction
        prediction_counts = Counter(predictions)
        most_common_count = prediction_counts.most_common(1)[0][1]
        
        # Return ratio of most common to total
        return most_common_count / len(predictions)
    
    def _get_consensus_prediction(self, predictions: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Get the most common prediction from a list."""
        if not predictions:
            return "NA", "NA"
        
        # Simple majority vote
        prediction_counts = Counter(predictions)
        return prediction_counts.most_common(1)[0][0]
    
    def _get_cok_system_prompt(self) -> str:
        """Get the system prompt for CoK reasoning."""
        return """You are an expert cooking assistant with access to culinary knowledge.
You understand recipe instructions and can reason about inputs and outputs using cooking knowledge.

Key principles:
- Only include comestibles (food items) in inputs/outputs, never tools or locations
- Be specific about ingredient states and transformations
- Use knowledge to make accurate predictions about cooking processes
- Consider chemical and physical transformations in cooking
- Track how ingredients flow through the recipe steps"""
    
    def _build_context(self, recipe: Recipe, up_to_step: int) -> List[Dict[str, str]]:
        """Build context from previous steps."""
        context = []
        
        # Add ingredients
        context.append({
            "type": "ingredients",
            "content": ", ".join(recipe.ingredients) if recipe.ingredients else "Not specified"
        })
        
        # Add previous steps
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
    
    def _format_previous_steps(self, recipe: Recipe, up_to_step: int) -> str:
        """Format previous steps for prompt."""
        if up_to_step == 0:
            return "No previous steps."
        
        lines = []
        for i in range(up_to_step):
            step = recipe.steps[i]
            lines.append(f"Step {i+1}: {step.instruction}")
            if step.input_pred:
                lines.append(f"  Input: {step.input_pred}")
            if step.output_pred:
                lines.append(f"  Output: {step.output_pred}")
        
        return "\n".join(lines)
    
    def _format_knowledge(self, knowledge: List[Dict[str, str]]) -> str:
        """Format retrieved knowledge for prompt."""
        if not knowledge:
            return "No specific knowledge retrieved."
        
        lines = []
        for i, k in enumerate(knowledge[:5]):  # Limit to top 5
            lines.append(f"{i+1}. {k.get('content', 'N/A')}")
            if 'source' in k:
                lines.append(f"   Source: {k['source']}")
        
        return "\n".join(lines)
    
    def _parse_cok_prediction(self, response: str) -> Tuple[str, str]:
        """Parse CoK model response to extract input and output."""
        lines = response.strip().split('\n')
        pred_input = "NA"
        pred_output = "NA"
        
        for line in lines:
            line_lower = line.lower()
            if 'input:' in line_lower:
                pred_input = line.split(':', 1)[1].strip()
            elif 'output:' in line_lower:
                pred_output = line.split(':', 1)[1].strip()
        
        # Clean up formatting
        pred_input = pred_input.strip('[]()').strip()
        pred_output = pred_output.strip('[]()').strip()
        
        return pred_input, pred_output