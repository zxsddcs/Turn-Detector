
import os
import json
import random
from tqdm import tqdm
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype="auto", device_map="auto")


def filter_data_with_llm(instr: str) -> str:
    prompt = f"""
        Ask the model to judge whether the sentence is valid or not
        If the sentence is valid, output 1, otherwise output 0
        Input: {instr}
        Output:
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        model_inputs = tokenizer([inputs], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=20,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return None


def add_oral_words(trunc_instr: str, language: str) -> str:
    ORAL_WORDS = {
        "en": ['uh', 'um', 'ah', 'er', 'hmm'],  # Ask LLM for more words
        "zh": ['嗯', '啊', '呃', '那个', '对吧']
    }
    filler = random.choice(ORAL_WORDS[language])
    return f"{trunc_instr} {filler}"


def filter_punctuation(instruction: str) -> str:
    PUNCTUATION = ['，','。','！','？','：', '、', '\n',',', '.', '!', '?', ':']
    for punc in PUNCTUATION:
        instruction = instruction.replace(punc, " ")
    return instruction.strip()
    

def process_dataset(dataset: list, language: str) -> list:
    data = []
    for sample in dataset:
        instr = sample['instruction']
        response = filter_data_with_llm(instr)
        if response == "1" and len(instr) > 10:
            words = instr.split(" ")
            if len(words) > 1:
                # Cut the sentence into two parts
                max_cut = max(1, len(words) // 2)  # Ensure at least 1
                cut = random.randint(1, max_cut)
                front_instr = " ".join(words[:cut])
                tail_instr = " ".join(words[cut:])
                # Truncate and randomly add 1 filler word
                data.append({
                    "instruction": f"{add_oral_words(front_instr, language)} ... {tail_instr}",
                    "input": "",
                    "output": "<|im_end|>"
                })
                data.append({
                    "instruction": f"{add_oral_words(front_instr, language)} {tail_instr}",
                    "input": "",
                    "output": "<|im_end|>"
                })
            data.append({
                "instruction": instr,
                "input": "",
                "output": "<|im_end|>"
            })
    return data


def clean_eos_token(dataset: list) -> list:
    for sample in dataset:
        sample["output"] = sample["output"].replace("<|im_end|>", "")
    return dataset


def format_datasets(dataset, language):
    """
    Format the dataset:
        1. Merge instruction and input
        2. Remove common punctuation at the end of the sentence
        3. Optimize the dataset, insert filler words
    """
    data = []
    for sample in tqdm(dataset):
        instr = f"{sample['instruction']} {sample['input']}"
        instr = filter_punctuation(instr)
        data.append({
            "instruction": instr,
            "input": "",
            "output": ""
        })
    data = process_dataset(data, language)
    # random.shuffle(data)
    # Separate training set/test set
    if len(data) > 2000:
        train_dataset, test_dataset = data[:-1000], data[-1000:]
    else:
        train_dataset, test_dataset = data[:1000], data[1000:]
    train_dataset = clean_eos_token(train_dataset)
    print(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")
    return train_dataset, test_dataset


def save_datasets(datasets: List[Dict], path: str) -> bool:
    """Save datasets to JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(datasets, file, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {path}, dataset length: {len(datasets)}")
    except Exception as e:
        print(f"Failed to save dataset: {e}")


def main():
    """Main function with improved configuration and error handling."""
    config = {
        "base_path": "base_path",
        "output_path": "output_path",
        "datasets": [
            {"language": "en", "path": "dataset.json"},
            {"language": "zh", "path": "dataset.json"},
        ]
    }
    try:
        os.makedirs(config["output_path"], exist_ok=True)
        for dataset in config["datasets"]:
            input_path = os.path.join(config["base_path"], dataset["path"])
            train_path = os.path.join(config["output_path"], f"train_{dataset['path'].split('/')[-1]}")
            test_path = os.path.join(config["output_path"], f"test_{dataset['path'].split('/')[-1]}")

            with open(input_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            train_dataset, test_dataset = format_datasets(data, dataset["language"])
            save_datasets(train_dataset, train_path)
            save_datasets(test_dataset, test_path)
            print(f"Processed dataset: {dataset['path']}")
    except Exception as e:
        print(f"Failed to process dataset: {e}")


if __name__ == "__main__":
    main()