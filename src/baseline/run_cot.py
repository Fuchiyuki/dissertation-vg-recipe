import os
import glob
import json
import openai

def load_instructions(data_dir):
    """
    Load all recipe files (*.txt) in data_dir and extract each instruction step.
    Returns a list of dicts with keys: file, step_index, instruction.
    """
    instructions = []
    for filepath in glob.glob(os.path.join(data_dir, '*.txt')):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for idx, step in enumerate(data.get('table', [])):
                instructions.append({
                    'file': os.path.basename(filepath),
                    'step_index': idx,
                    'instruction': step.get('instructions', '').strip()
                })
    return instructions


def build_prompt(inst_text):
    """
    Construct a Chain-of-Thought prompt for triple extraction.
    """
    prompt = f"""
You are an AI assistant that extracts structured knowledge from cooking instructions.
Think step-by-step, then provide the final relation triple in the form <subject, relation, object>.

Instruction: "{inst_text}"

Let's think step by step:
1.
2.
...
Based on the above reasoning, the triple is:"""
    return prompt


def extract_triple(entry, model="gpt-4-turbo"):  
    """
    Call the OpenAI ChatCompletion API with a CoT prompt to extract a triple.
    Returns the assistant's full response (chain-of-thought + triple).
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = build_prompt(entry['instruction'])
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI that performs detailed step-by-step reasoning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()


def main():
    data_dir = 'data/train'
    out_path = 'outputs/cot_results.json'

    # Load instructions
    instructions = load_instructions(data_dir)

    # Extract triples with CoT
    results = []
    for entry in instructions:
        cot_output = extract_triple(entry)
        results.append({**entry, 'cot_output': cot_output})
        print(f"Processed {entry['file']} step {entry['step_index']}")

    # Save results
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"CoT extraction completed. Results saved to {out_path}.")

if __name__ == '__main__':
    main()
