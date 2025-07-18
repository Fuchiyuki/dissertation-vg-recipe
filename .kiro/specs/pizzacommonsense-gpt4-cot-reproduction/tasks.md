# Implementation Plan

- [x] 1. Create Google Colab notebook structure and setup
  - Create a new Jupyter notebook with proper title and description
  - Add markdown cells for each major section with clear explanations
  - Structure the notebook into 6 main sections as defined in the design
  - _Requirements: 2.1, 8.1_

- [x] 2. Implement environment setup and dependency management
  - Create library installation cell with all required packages (openai>=1.30, pandas, tqdm, rouge-score, evaluate, tiktoken)
  - Add secure API key input functionality using getpass
  - Include import statements for all necessary modules
  - _Requirements: 2.1, 2.2_

- [ ] 3. Implement data loading functionality
- [x] 3.1 Create data folder iterator for PizzaCommonsense dataset
  - Write function to traverse data/PizzaCommonsense/train and data/PizzaCommonsense/val folders
  - Implement JSON file reading and parsing for recipe files
  - Create iterator that yields individual recipe steps from the "table" field
  - _Requirements: 3.1, 3.2_

- [x] 3.2 Add data validation and error handling
  - Validate required fields exist in each recipe step (instructions, actions, input, output)
  - Handle malformed JSON files gracefully with logging
  - Skip corrupted entries and continue processing
  - _Requirements: 3.3_

- [ ] 4. Implement Chain-of-Thought prompt generation
- [x] 4.1 Create CoT prompt template system
  - Define system message for cooking reasoning assistant
  - Create prompt formatting function that includes instructions, actions, and CoT trigger
  - Implement "Let's think step by step" methodology
  - _Requirements: 4.1, 4.2_

- [x] 4.2 Add prompt validation and testing
  - Test prompt generation with sample recipe steps
  - Validate prompt format matches expected structure
  - Ensure proper escaping of special characters in recipe text
  - _Requirements: 4.1_

- [ ] 5. Implement GPT-4 Turbo inference system
- [x] 5.1 Create OpenAI API client and prediction function
  - Initialize OpenAI client with API key
  - Implement GPT-4 prediction function with proper error handling
  - Use gpt-4o-mini model for cost-effective inference
  - Set temperature=0 for consistent results
  - _Requirements: 4.2, 6.3_

- [x] 5.2 Implement response parsing for input/output extraction
  - Create regex patterns to extract predicted input and output from model responses
  - Handle cases where parsing fails with fallback empty strings
  - Normalize extracted text for consistent formatting
  - _Requirements: 4.3_

- [x] 5.3 Add rate limiting and retry logic
  - Implement configurable delays between API calls (default 0.5s)
  - Add exponential backoff for rate limit errors
  - Retry failed requests up to 3 times with increasing delays
  - _Requirements: 6.1, 6.3_

- [ ] 6. Implement batch processing system
- [x] 6.1 Create main prediction loop with progress tracking
  - Iterate through validation split data using tqdm for progress display
  - Generate predictions for each recipe step
  - Collect results in structured format for evaluation
  - _Requirements: 6.2, 7.2_

- [x] 6.2 Add result storage and CSV export
  - Store predictions alongside ground truth in pandas DataFrame
  - Export results to CSV file for further analysis
  - Include all relevant fields (instructions, actions, ground truth, predictions)
  - _Requirements: 7.2_

- [ ] 7. Implement comprehensive evaluation metrics
- [x] 7.1 Create Exact Match Accuracy (EMA) calculation
  - Implement EMA for both input and output predictions
  - Normalize strings for case and whitespace differences
  - Calculate average EMA score across input and output
  - _Requirements: 5.1, 5.4_

- [x] 7.2 Implement Rouge-L score calculation
  - Use evaluate library to compute Rouge-L scores
  - Calculate separate scores for input and output predictions
  - Handle edge cases where predictions or references are empty
  - _Requirements: 5.2_

- [x] 7.3 Add BERTScore evaluation
  - Implement BERTScore calculation using evaluate library
  - Focus on output predictions for semantic similarity
  - Configure for English language evaluation
  - _Requirements: 5.3_

- [ ] 8. Create results reporting and benchmark comparison
- [x] 8.1 Format metric results for comparison
  - Display results in clear, readable format
  - Create comparison table with paper benchmark values
  - Highlight metrics that meet or exceed paper performance
  - _Requirements: 7.3, 8.3_

- [x] 8.2 Add detailed analysis and interpretation
  - Provide markdown cells explaining metric meanings
  - Include cost estimation and runtime information
  - Add suggestions for potential improvements
  - _Requirements: 8.1, 8.3_

- [ ] 9. Add extensibility framework for advanced techniques
- [x] 9.1 Structure code for Self-Consistency extension
  - Design prediction function to support multiple generations
  - Add placeholder for majority voting implementation
  - Document how to enable Self-Consistency mode
  - _Requirements: 7.1_

- [x] 9.2 Prepare Few-Shot CoT framework
  - Add function to sample exemplars from training data
  - Create template for few-shot prompt formatting
  - Document how to enable few-shot mode
  - _Requirements: 7.1_

- [ ] 10. Add comprehensive documentation and instructions
- [x] 10.1 Create setup and usage documentation
  - Add markdown cells explaining each section's purpose
  - Include step-by-step execution instructions
  - Document expected runtime and cost estimates
  - _Requirements: 8.1, 8.2_

- [x] 10.2 Add troubleshooting and FAQ section
  - Document common issues and solutions
  - Include API key setup instructions
  - Add guidance for interpreting results
  - _Requirements: 8.1_