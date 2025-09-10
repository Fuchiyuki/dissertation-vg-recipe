# Directory: src/baseline/qwen/

# prompt_template.py
from typing import List

def build_cot_prompt(instruction: str) -> str:
    """
    Compose a detailed Chain-of-Thought prompt for extracting a single knowledge triple from cooking instructions.
    The prompt instructs the model to think step-by-step, articulate each reasoning step clearly,
    and then provide the final relation triple in the format <subject, relation, object>.
    """
    return f"""
You are an AI assistant specialized in extracting structured knowledge from cooking instructions.
For the following single instruction step, perform the following tasks:

1. Carefully analyze the instruction and identify all implicit and explicit actions, entities, and effects.
2. Write out your reasoning in a clear, numbered list of thought steps, ensuring each step builds on the previous one.
3. After completing the reasoning steps, provide exactly one relation triple summarizing the core knowledge in the format <subject, relation, object>.

**Important:**
- Do not include any text outside the numbered reasoning steps and the final triple.
- Ensure your reasoning is thorough, referencing any implicit assumptions or domain knowledge.

Instruction:
"{instruction}"

Let's think step by step:
1.
2.
...

Based on the above reasoning, the triple is: <subject, relation, object>
""".strip()


# run_cot.py
import os, glob, json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from prompt_template import build_cot_prompt

def init_generator(model_name: str, max_new_tokens: int = 256, temperature: float = 0.7):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    gen = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens, temperature=temperature
    )
    return gen

MODEL_NAME = "qwen/Qwen-2.5"
generator = init_generator(MODEL_NAME)

def load_instructions(data_dir: str):
    instructions = []
    for path in glob.glob(os.path.join(data_dir, '*.txt')):
        data = json.load(open(path, encoding='utf-8'))
        for idx, step in enumerate(data.get('table', [])):
            instructions.append({
                'file': os.path.basename(path),
                'step': idx,
                'text': step.get('instructions','').strip()
            })
    return instructions


def main_cot():
    inputs = load_instructions('data/train')
    outputs = []
    for entry in inputs:
        prompt = build_cot_prompt(entry['text'])
        resp = generator(prompt)[0]['generated_text']
        outputs.append({**entry, 'cot_output': resp})
        print(f"[CoT] {entry['file']}#{entry['step']} done")
    os.makedirs('outputs/baseline/qwen', exist_ok=True)
    with open('outputs/baseline/qwen/cot_results.json','w',encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main_cot()


# run_cok.py
import os, glob, json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from prompt_template import build_cok_prompt

def init_generator(model_name: str, max_new_tokens: int = 256, temperature: float = 0.0):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    gen = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=max_new_tokens, temperature=temperature
    )
    return gen

MODEL_NAME = "qwen/Qwen-2.5"
generator = init_generator(MODEL_NAME)


def main_cok(n_triples: int = 3):
    inputs = load_instructions('data/train')
    outputs = []
    for entry in inputs:
        prompt = build_cok_prompt(entry['text'], num_triples=n_triples)
        resp = generator(prompt)[0]['generated_text']
        outputs.append({**entry, 'cok_output': resp})
        print(f"[CoK] {entry['file']}#{entry['step']} done")
    os.makedirs('outputs/baseline/qwen', exist_ok=True)
    with open('outputs/baseline/qwen/cok_results.json','w',encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main_cok()

