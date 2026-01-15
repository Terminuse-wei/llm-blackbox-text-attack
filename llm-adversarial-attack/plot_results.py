import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/comparison.csv")

plt.figure()
plt.bar(df["Method"], df["SuccessRate"])
plt.xlabel("Attack Method")
plt.ylabel("Success Rate")
plt.title("LLM Prompt-based Adversarial Attack Performance")
plt.savefig("results/success_rate.png")
print("Saved results/success_rate.png")
