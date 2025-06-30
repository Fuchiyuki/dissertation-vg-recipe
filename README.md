cok-pizzacommonsense/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config.py                   # Configuration and API keys
│
├── data/                       # Dataset directory
│   ├── download_data.py        # Script to download PizzaCommonsense
│   ├── preprocess.py           # Data preprocessing utilities
│   └── pizzacommonsense/       # Downloaded dataset files
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── models/                 # Model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py       # Abstract base class
│   │   ├── cot_model.py        # Chain-of-Thought implementation
│   │   └── cok_model.py        # Chain-of-Knowledge implementation
│   │
│   ├── knowledge/              # Knowledge retrieval components
│   │   ├── __init__.py
│   │   ├── query_generator.py  # Generate queries for knowledge sources
│   │   ├── knowledge_sources.py # Knowledge source interfaces
│   │   └── retriever.py        # Knowledge retrieval logic
│   │
│   ├── prompts/                # Prompt templates
│   │   ├── __init__.py
│   │   ├── cot_prompts.py      # CoT prompt templates
│   │   └── cok_prompts.py      # CoK prompt templates
│   │
│   └── evaluation/             # Evaluation metrics
│       ├── __init__.py
│       ├── metrics.py          # Evaluation metrics implementation
│       └── evaluator.py        # Main evaluation logic
│
├── experiments/                # Experiment scripts
│   ├── run_cot_baseline.py     # Run CoT baseline
│   ├── run_cok.py              # Run CoK method
│   └── compare_methods.py      # Compare results
│
├── results/                    # Experiment results
│   ├── cot_results/
│   └── cok_results/
│
└── notebooks/                  # Jupyter notebooks for analysis
    ├── data_exploration.ipynb
    └── results_analysis.ipynb
