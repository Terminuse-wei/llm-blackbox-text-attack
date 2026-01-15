import json, os, re, time
import requests
from transformers import pipeline
from datasets import load_dataset

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")

PROMPTS = {
    "deepwordbug": "prompts/deepwordbug.txt",
    "textfooler": "prompts/textfooler.txt",
    "pwws": "prompts/pwws.txt",
}

def call_ollama(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

def clean_one_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```.*?\n|\n```$", "", text, flags=re.S)
    first = text.splitlines()[0].strip()
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def pick_recipes(orig_conf: float, length: int):
    # 简单策略：高置信度先上词级（更强），长度短优先字符级
    if orig_conf >= 0.95:
        return ["textfooler", "pwws", "deepwordbug"]
    if length <= 8:
        return ["deepwordbug", "textfooler", "pwws"]
    return ["pwws", "textfooler", "deepwordbug"]

def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/llm_attacks.jsonl"

    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # 先用 SST-2 validation[:20] 做小实验（你可以改成 50）
    ds = load_dataset("glue", "sst2", split="validation[:20]")

    prompts_cache = {k: load_prompt(v) for k, v in PROMPTS.items()}

    for idx, item in enumerate(ds):
        text = item["sentence"]
        orig = clf(text)[0]
        orig_label = orig["label"]
        orig_conf = float(orig["score"])
        length = len(text.split())

        recipes = pick_recipes(orig_conf, length)
        success = False

        for attempt, recipe in enumerate(recipes, start=1):
            tmpl = prompts_cache[recipe]
            prompt = tmpl.format(input_text=text, label=orig_label, confidence=f"{orig_conf:.4f}")

            llm_raw = call_ollama(prompt)
            adv_text = clean_one_sentence(llm_raw)
            adv = clf(adv_text)[0]
            flipped = adv["label"] != orig_label

            record = {
                "idx": idx,
                "attempt": attempt,
                "recipe": recipe,
                "input_text": text,
                "orig_pred": orig,
                "adv_text": adv_text,
                "adv_pred": adv,
                "flipped": flipped,
                "ollama_model": OLLAMA_MODEL,
                "ts": time.time(),
            }

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[{idx}] attempt {attempt}/3 recipe={recipe} flipped={flipped} orig={orig_label}({orig_conf:.3f}) -> adv={adv['label']}({adv['score']:.3f})")

            if flipped:
                success = True
                break

        if not success:
            print(f"[{idx}] no flip after 3 attempts")

    print(f"Saved logs to {out_path}")

if __name__ == "__main__":
    main()
