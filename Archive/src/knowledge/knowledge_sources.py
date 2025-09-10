# src/knowledge/knowledge_sources.py

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import json
from pathlib import Path


class KnowledgeSource(ABC):
    """Abstract base class for knowledge sources."""
    
    @abstractmethod
    def search(self, query: str) -> List[Dict[str, str]]:
        """Search the knowledge source with a query."""
        pass


class CookingTechniquesKB(KnowledgeSource):
    """Knowledge base for cooking techniques and transformations."""
    
    def __init__(self):
        # In a real implementation, this would load from a file or database
        self.knowledge = [
            {
                "technique": "mixing",
                "description": "Combining ingredients uniformly",
                "common_results": {
                    "flour + water + yeast": "dough",
                    "eggs + milk + flour": "batter",
                    "oil + vinegar": "vinaigrette",
                    "tomatoes + herbs + oil": "tomato sauce"
                }
            },
            {
                "technique": "baking",
                "description": "Cooking with dry heat in an oven",
                "transformations": {
                    "dough": "bread/pizza crust",
                    "batter": "cake/muffins",
                    "raw pizza": "baked pizza"
                }
            },
            {
                "technique": "sautéing",
                "description": "Cooking quickly in a small amount of fat",
                "transformations": {
                    "onions": "sautéed onions",
                    "garlic": "sautéed garlic",
                    "vegetables": "sautéed vegetables"
                }
            },
            {
                "technique": "spreading",
                "description": "Distributing a substance over a surface",
                "common_applications": {
                    "sauce on dough": "sauced pizza base",
                    "cheese on pizza": "cheese-topped pizza"
                }
            }
        ]
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """Search for relevant cooking techniques."""
        results = []
        query_lower = query.lower()
        
        for item in self.knowledge:
            if item["technique"] in query_lower or query_lower in item["description"].lower():
                results.append({
                    "content": f"{item['technique']}: {item['description']}",
                    "source": "cooking_techniques_kb",
                    "details": item
                })
        
        return results


class IngredientPropertiesKB(KnowledgeSource):
    """Knowledge base for ingredient properties and combinations."""
    
    def __init__(self):
        self.knowledge = [
            {
                "ingredient": "yeast",
                "properties": ["leavening agent", "requires warm water to activate"],
                "common_uses": ["bread", "pizza dough", "fermentation"]
            },
            {
                "ingredient": "mozzarella",
                "properties": ["melts well", "stretchy when melted", "mild flavor"],
                "states": ["shredded", "sliced", "fresh", "melted"]
            },
            {
                "ingredient": "tomato sauce",
                "components": ["tomatoes", "herbs", "oil", "seasonings"],
                "variations": ["marinara", "pizza sauce", "seasoned tomato sauce"]
            },
            {
                "combination": "pizza assembly",
                "order": ["dough base", "sauce layer", "cheese layer", "toppings"],
                "result": "assembled pizza"
            }
        ]
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """Search for ingredient-related knowledge."""
        results = []
        query_lower = query.lower()
        
        for item in self.knowledge:
            if any(key in query_lower for key in item.keys()):
                results.append({
                    "content": json.dumps(item, indent=2),
                    "source": "ingredient_properties_kb"
                })
        
        return results


class CookingCommonsenseKB(KnowledgeSource):
    """General cooking commonsense knowledge."""
    
    def __init__(self):
        self.rules = [
            "When 'seasoning' is mentioned, it typically includes salt and pepper",
            "Preheating the oven doesn't involve food items (use NA for input/output)",
            "Kneading dough results in smooth, elastic dough",
            "Rising/proofing dough results in risen/expanded dough",
            "Melted cheese becomes 'melted cheese', not just 'cheese'",
            "Chopped vegetables should specify the cut (diced, sliced, minced)",
            "Mixing dry and wet ingredients separately is common in baking",
            "Oil or butter is often an implicit ingredient for sautéing/frying"
        ]
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """Search for relevant commonsense rules."""
        results = []
        query_lower = query.lower()
        
        for rule in self.rules:
            if any(word in rule.lower() for word in query_lower.split()):
                results.append({
                    "content": rule,
                    "source": "cooking_commonsense"
                })
        
        return results[:3]  # Limit to top 3 most relevant


class CookingKnowledgeBase:
    """Aggregates all cooking knowledge sources."""
    
    def __init__(self):
        self.sources = [
            CookingTechniquesKB(),
            IngredientPropertiesKB(),
            CookingCommonsenseKB()
        ]
    
    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """Retrieve knowledge from all sources."""
        all_results = []
        
        for source in self.sources:
            results = source.search(query)
            all_results.extend(results)
        
        # Sort by relevance (simple heuristic: longer content = more detailed)
        all_results.sort(key=lambda x: len(x.get('content', '')), reverse=True)
        
        return all_results[:5]  # Return top 5 results