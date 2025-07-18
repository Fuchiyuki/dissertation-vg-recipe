# Design Document

## Overview

This design outlines the implementation of a Google Colab notebook for reproducing PizzaCommonSense dataset experiments using GPT-4 Turbo with Chain-of-Thought prompting. The system will process recipe steps, generate predictions using CoT methodology, and evaluate results against published benchmarks.

## Architecture

The notebook follows a linear execution flow with six main sections:

1. **Setup & Configuration**: Environment preparation and API key management
2. **Data Processing**: Folder traversal and JSON parsing
3. **Prompt Engineering**: CoT template generation and formatting
4. **Model Inference**: GPT-4 Turbo API calls with rate limiting
5. **Batch Processing**: Automated prediction generation for test split
6. **Evaluation & Reporting**: Metric calculation and benchmark comparison

## Components and Interfaces

### Data Loader Component

```python
class PizzaDataLoader:
    def __init__(self, data_path: str)
    def iter_tables(self, split: str) -> Iterator[Dict]
```

**Purpose**: Handles folder traversal and JSON parsing for PizzaCommonSense dataset
**Input**: Data folder path and dataset split name (train/val)
**Output**: Iterator yielding recipe step dictionaries

### Prompt Generator Component

```python
class CoTPromptGenerator:
    def __init__(self, system_message: str)
    def make_prompt(self, row: Dict) -> str
```

**Purpose**: Creates Chain-of-Thought prompts from recipe step data
**Input**: Recipe step dictionary with instructions and actions
**Output**: Formatted prompt string with CoT trigger

### GPT-4 Inference Component

```python
class GPT4Predictor:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini")
    def predict(self, prompt: str) -> str
    def parse_io(self, response: str) -> Tuple[str, str]
```

**Purpose**: Handles OpenAI API calls and response parsing
**Input**: Formatted prompt string
**Output**: Parsed input and output predictions

### Evaluation Component

```python
class MetricsCalculator:
    def calculate_ema(self, predictions: List, references: List) -> float
    def calculate_rouge_l(self, predictions: List, references: List) -> float
    def calculate_bertscore(self, predictions: List, references: List) -> float
```

**Purpose**: Computes evaluation metrics for model predictions
**Input**: Lists of predictions and ground truth references
**Output**: Metric scores (EMA, Rouge-L, BERTScore)

## Data Models

### Recipe Step Schema
```json
{
    "instructions": "string - cooking instruction text",
    "actions": "string - specific action being performed", 
    "input": "string - input ingredients/items",
    "output": "string - resulting ingredients/items"
}
```

### Prediction Record Schema
```json
{
    "instructions": "string - original instruction",
    "actions": "string - original action",
    "input": "string - ground truth input",
    "output": "string - ground truth output",
    "pred_input": "string - predicted input",
    "pred_output": "string - predicted output"
}
```

### Metrics Schema
```json
{
    "ema_input": "float - exact match accuracy for inputs",
    "ema_output": "float - exact match accuracy for outputs", 
    "ema_average": "float - average EMA score",
    "rouge_l_input": "float - Rouge-L score for inputs",
    "rouge_l_output": "float - Rouge-L score for outputs",
    "bertscore_f1": "float - BERTScore F1 for outputs"
}
```

## Error Handling

### API Rate Limiting
- Implement exponential backoff for rate limit errors
- Add configurable delays between API calls (default 0.5s)
- Retry failed requests up to 3 times with increasing delays

### Data Processing Errors
- Handle malformed JSON files gracefully with logging
- Skip corrupted entries and continue processing
- Validate required fields exist before processing

### Parsing Errors
- Use flexible regex patterns for input/output extraction
- Provide fallback empty strings for unparseable responses
- Log parsing failures for debugging

## Testing Strategy

### Unit Testing Approach
- Test data loader with sample data folders
- Validate prompt generation with known inputs
- Mock API responses for inference testing
- Verify metric calculations with known ground truth

### Integration Testing
- End-to-end test with small dataset subset
- Validate API key handling and authentication
- Test error recovery and retry mechanisms
- Verify output file generation and format

### Performance Testing
- Measure API call latency and throughput
- Monitor memory usage during batch processing
- Test with full dataset size for scalability
- Validate rate limiting effectiveness

## Implementation Details

### Notebook Structure
The Colab notebook will be organized into clearly marked sections:

1. **Setup Cell**: Library installation and imports
2. **Configuration Cell**: API key input and environment setup
3. **Data Loading Cell**: Folder traversal and iterator setup
4. **Prompt Engineering Cell**: CoT template definition
5. **Inference Cell**: GPT-4 prediction function
6. **Batch Processing Cell**: Main execution loop with progress tracking
7. **Evaluation Cells**: Metric calculation and reporting
8. **Results Cell**: Comparison with paper benchmarks

### Prompt Template Design
```
System: You are a cooking-reasoning assistant. Given an instruction and an action, predict the input comestibles and the output comestible/result for that step.

User: Instruction: {instructions}
Action: {actions}
Let's think step by step.
```

### Response Parsing Strategy
Use regex patterns to extract structured predictions:
- `Input\s*[:：]\s*(.+)` for input extraction
- `Output\s*[:：]\s*(.+)` for output extraction
- Fallback to empty strings if patterns don't match

### Rate Limiting Implementation
- Base delay of 0.5 seconds between requests
- Exponential backoff: 1s, 2s, 4s for retries
- Maximum 3 retry attempts per request
- Progress tracking with tqdm for user feedback

## Extension Points

### Self-Consistency Enhancement
- Generate multiple predictions with temperature > 0
- Implement majority voting for final predictions
- Compare single vs. ensemble performance

### Few-Shot CoT Implementation
- Sample exemplars from training split
- Format examples in prompt template
- Evaluate impact of different shot counts (1, 3, 5)

### Advanced Prompt Engineering
- Experiment with different CoT trigger phrases
- Add explicit output format instructions
- Test domain-specific reasoning prompts

## Performance Considerations

### Memory Management
- Process data in streaming fashion to avoid loading entire dataset
- Use generators for large file iteration
- Clear intermediate variables after processing

### API Cost Optimization
- Use gpt-4o-mini for cost-effective inference
- Implement token counting for cost estimation
- Provide cost warnings for large datasets

### Execution Time
- Parallel processing not recommended due to rate limits
- Estimated runtime: ~30-60 minutes for full test split
- Progress indicators for long-running operations