# Requirements Document

## Introduction

This feature implements a Google Colab notebook for reproducing PizzaCommonSense dataset experiments using GPT-4 Turbo with Chain-of-Thought (CoT) prompting. The goal is to achieve performance metrics equal to or better than the published paper results (GPT-4 + CoT: EMA 26.7%, RougeLin 51.4, RougeLout 52.3) on the test split input/output prediction task.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to reproduce the PizzaCommonSense GPT-4 CoT experiment in Google Colab, so that I can validate the published results and potentially improve upon them.

#### Acceptance Criteria

1. WHEN the notebook is executed THEN the system SHALL achieve at least one metric score greater than or equal to the paper's published values
2. WHEN processing the test split THEN the system SHALL predict both input and output for each recipe step
3. WHEN using GPT-4 Turbo THEN the system SHALL implement Chain-of-Thought prompting methodology

### Requirement 2

**User Story:** As a user, I want a complete Colab notebook setup, so that I can run the experiment without additional configuration.

#### Acceptance Criteria

1. WHEN setting up the environment THEN the system SHALL install all required libraries (openai>=1.30, pandas, tqdm, rouge-score, evaluate, tiktoken)
2. WHEN configuring API access THEN the system SHALL securely prompt for OpenAI API key input
3. WHEN running on Colab THEN the system SHALL work without GPU requirements (CPU-only inference)

### Requirement 3

**User Story:** As a researcher, I want to process the PizzaCommonSense dataset efficiently, so that I can handle the complete validation split without manual intervention.

#### Acceptance Criteria

1. WHEN loading data THEN the system SHALL read from data/PizzaCommonsense folder structure with train and val subfolders
2. WHEN parsing JSON THEN the system SHALL extract instructions, actions, input, and output from each recipe step
3. WHEN iterating through data THEN the system SHALL process all validation split files automatically

### Requirement 4

**User Story:** As an experimenter, I want proper Chain-of-Thought prompting, so that the model can reason through cooking steps effectively.

#### Acceptance Criteria

1. WHEN generating prompts THEN the system SHALL include "Let's think step by step" CoT trigger
2. WHEN calling GPT-4 THEN the system SHALL use appropriate system message for cooking reasoning
3. WHEN processing responses THEN the system SHALL parse predicted input and output from model text

### Requirement 5

**User Story:** As a researcher, I want comprehensive evaluation metrics, so that I can compare results against the published paper.

#### Acceptance Criteria

1. WHEN calculating EMA THEN the system SHALL compute exact match accuracy for both input and output predictions
2. WHEN computing Rouge-L THEN the system SHALL calculate scores for both input and output separately
3. WHEN evaluating results THEN the system SHALL provide BERTScore as an additional semantic similarity metric
4. WHEN comparing strings THEN the system SHALL normalize for case and whitespace differences

### Requirement 6

**User Story:** As a user, I want rate limiting and cost management, so that I can run experiments safely within API limits.

#### Acceptance Criteria

1. WHEN making API calls THEN the system SHALL implement appropriate delays between requests
2. WHEN processing large datasets THEN the system SHALL provide progress tracking with tqdm
3. WHEN handling errors THEN the system SHALL implement basic retry logic for API failures

### Requirement 7

**User Story:** As a researcher, I want extensible experiment framework, so that I can easily add additional techniques like Self-Consistency or Few-Shot CoT.

#### Acceptance Criteria

1. WHEN implementing base functionality THEN the system SHALL structure code to support additional prompting strategies
2. WHEN saving results THEN the system SHALL export predictions to CSV for further analysis
3. WHEN reporting metrics THEN the system SHALL format results for easy comparison with paper values

### Requirement 8

**User Story:** As a user, I want clear documentation and instructions, so that I can understand and modify the experiment setup.

#### Acceptance Criteria

1. WHEN viewing the notebook THEN the system SHALL provide markdown cells explaining each major section
2. WHEN setting up data THEN the system SHALL include instructions for accessing the existing data folder structure
3. WHEN interpreting results THEN the system SHALL provide comparison tables with paper benchmarks