import json, csv
from collections import defaultdict

records = []
with open("results/llm_attacks.jsonl","r",encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

by_sample = defaultdict(list)
for r in records:
    by_sample[r["idx"]].append(r)

summary = {}
attempts = defaultdict(int)
success = defaultdict(set)

for idx, lst in by_sample.items():
    for r in lst:
        attempts[r["recipe"]] += 1
        if r["flipped"]:
            success[r["recipe"]].add(idx)

methods = sorted(attempts.keys())

rows = []
for m in methods:
    total_samples = len(by_sample)
    succ = len(success[m])
    rate = succ / total_samples
    avg_tries = attempts[m] / total_samples
    rows.append([m, rate, avg_tries, "Prompt-based LLM attack"])

with open("results/comparison.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["Method","SuccessRate","AvgAttempts","Notes"])
    w.writerows(rows)

print("Saved results/comparison.csv")
