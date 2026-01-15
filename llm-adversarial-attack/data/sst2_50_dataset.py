from datasets import load_dataset
from textattack.datasets import Dataset

# TextAttack expects a textattack.datasets.Dataset instance named `dataset`
hf = load_dataset("glue", "sst2", split="validation[:50]")
data = [(x["sentence"], x["label"]) for x in hf]

dataset = Dataset(data)
