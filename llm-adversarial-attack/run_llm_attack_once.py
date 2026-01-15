import json
import os
import re
import requests
from transformers import pipeline

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct")  # 你可改成你本地实际名字
PROMPT_PATH = os.environ.get("PROMPT_PATH", "prompts/deepwordbug.txt")

def call_ollama(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    text = r.json().get("response", "")
    return text.strip()

def extract_one_sentence(text: str) -> str:
    # 只要第一行/第一句，避免模型输出解释
    text = text.strip()
    text = re.sub(r"^```.*?\n|\n```$", "", text, flags=re.S)
    first_line = text.splitlines()[0].strip()
    # 去掉引号
    if (first_line.startswith('"') and first_line.endswith('"')) or (first_line.startswith("'") and first_line.endswith("'")):
        first_line = first_line[1:-1].strip()
    return first_line

def main():
    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    input_text = os.environ.get("INPUT_TEXT", "This movie was fantastic!")
    pred = clf(input_text)[0]
    label, confidence = pred["label"], float(pred["score"])

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        tmpl = f.read()

    prompt = tmpl.format(input_text=input_text, label=label, confidence=f"{confidence:.4f}")
    print("\n=== Original ===")
    print(input_text)
    print("Victim:", pred)

    print("\n=== LLM Prompt (truncated) ===")
    print(prompt[:500] + ("..." if len(prompt) > 500 else ""))

    llm_out = call_ollama(prompt)
    adv_text = extract_one_sentence(llm_out)

    adv_pred = clf(adv_text)[0]
    flipped = adv_pred["label"] != label

    print("\n=== Adversarial (LLM output) ===")
    print(adv_text)
    print("Victim:", adv_pred)
    print("\nFLIPPED?", flipped)

    os.makedirs("results", exist_ok=True)
    record = {
        "prompt_path": PROMPT_PATH,
        "input_text": input_text,
        "orig_pred": pred,
        "adv_text": adv_text,
        "adv_pred": adv_pred,
        "flipped": flipped,
        "ollama_model": OLLAMA_MODEL,
    }
    with open("results/llm_attack_once.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print("\nSaved: results/llm_attack_once.json")

if __name__ == "__main__":
    main()
