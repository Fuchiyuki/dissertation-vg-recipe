# Directory: src/baseline/qwen/

# prompt_template.py
from typing import List

def build_cot_prompt(instruction: str) -> str:
    """
    Compose a Chain-of-Thought prompt for triple extraction.
    """
    return f"""
You are an AI assistant that extracts structured knowledge from cooking instructions.
Think step-by-step, then provide the final relation triple in the form <subject, relation, object>.

Instruction: "{instruction}"

Let's think step by step:
1.
2.
...
Based on the above reasoning, the triple is:"""


def build_cok_prompt(instruction: str, num_triples: int = 3) -> str:
    """
    Compose a Chain-of-Knowledge prompt for extracting multiple knowledge triples.
    """
    header = (
        f"以下の料理手順をよく読んで、"  
        f"この手順に必要となる知識トリプルを〈主語, 関係, 目的語〉の形式で{num_triples}つ列挙してください。"
    )
    body = f"Instruction:\n\"{instruction}\""
    footer = (
        "出力フォーマット:\n"
        +"1. <subject_1, relation_1, object_1>\n"
        +"2. <subject_2, relation_2, object_2>\n"
        +"...\n"
        +f"{num_triples}. <subject_{num_triples}, relation_{num_triples}, object_{num_triples}>"
    )
    return "\n\n".join([header, body, footer])


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
