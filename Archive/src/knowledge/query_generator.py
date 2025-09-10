# src/knowledge/query_generator.py

from typing import List, Dict
import openai
from ..models.base_model import Recipe, RecipeStep


class QueryGenerator:
    """Generates queries for knowledge retrieval in CoK."""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def generate_queries(self, step: RecipeStep, recipe: Recipe) -> List[str]:
        """
        Generate queries for retrieving relevant knowledge about a cooking step.
        
        Args:
            step: The current cooking step
            recipe: The full recipe for context
            
        Returns:
            List of queries to search knowledge bases
        """
        queries = []
        
        # Query 1: Action-based query
        action_query = f"{step.action} cooking technique"
        queries.append(action_query)
        
        # Query 2: Instruction-based query
        # Extract key ingredients/items from instruction
        instruction_query = self._extract_key_terms(step.instruction)
        if instruction_query:
            queries.append(instruction_query)
        
        # Query 3: Transformation query
        transformation_query = f"what happens when you {step.action} {self._extract_object(step.instruction)}"
        queries.append(transformation_query)
        
        # Query 4: Context-aware query (if not first step)
        if recipe.steps.index(step) > 0:
            prev_step = recipe.steps[recipe.steps.index(step) - 1]
            if prev_step.output_pred:
                context_query = f"{step.action} {prev_step.output_pred}"
                queries.append(context_query)
        
        return queries
    
    def generate_structured_query(self, step: RecipeStep, query_type: str) -> Dict[str, str]:
        """
        Generate structured queries for specific knowledge sources.
        
        Args:
            step: The cooking step
            query_type: Type of query (e.g., 'ingredient', 'technique', 'transformation')
            
        Returns:
            Structured query dictionary
        """
        if query_type == "ingredient":
            return {
                "type": "ingredient_lookup",
                "terms": self._extract_ingredients(step.instruction),
                "action": step.action
            }
        elif query_type == "technique":
            return {
                "type": "technique_lookup",
                "action": step.action,
                "context": step.instruction
            }
        elif query_type == "transformation":
            return {
                "type": "transformation_lookup",
                "input": self._extract_object(step.instruction),
                "action": step.action,
                "expected_output": None
            }
        else:
            return {
                "type": "general",
                "query": step.instruction
            }
    
    def _extract_key_terms(self, instruction: str) -> str:
        """Extract key terms from instruction for query."""
        # Simple extraction - in production, use NLP
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'and', 'or'}
        words = instruction.lower().split()
        key_words = [w for w in words if w not in stopwords and len(w) > 2]
        return ' '.join(key_words[:3])  # Top 3 key words
    
    def _extract_object(self, instruction: str) -> str:
        """Extract the object being acted upon."""
        # Simple heuristic - get words after the action verb
        words = instruction.split()
        
        # Common cooking verbs
        cooking_verbs = {
            'mix', 'stir', 'bake', 'cook', 'fry', 'boil', 'chop', 
            'dice', 'slice', 'spread', 'add', 'combine', 'heat'
        }
        
        for i, word in enumerate(words):
            if word.lower() in cooking_verbs and i + 1 < len(words):
                # Return the next few words as the object
                return ' '.join(words[i+1:i+3])
        
        # Fallback: return last few words
        return ' '.join(words[-2:]) if len(words) > 1 else instruction
    
    def _extract_ingredients(self, instruction: str) -> List[str]:
        """Extract potential ingredients from instruction."""
        # This is a simplified version - in production, use NER
        common_ingredients = {
            'flour', 'water', 'yeast', 'salt', 'sugar', 'oil', 'butter',
            'cheese', 'mozzarella', 'tomato', 'sauce', 'pepper', 'garlic',
            'onion', 'herbs', 'dough', 'eggs', 'milk', 'cream'
        }
        
        words = instruction.lower().split()
        found_ingredients = []
        
        for word in words:
            # Remove punctuation
            clean_word = word.strip('.,!?;:')
            if clean_word in common_ingredients:
                found_ingredients.append(clean_word)
        
        return found_ingredients