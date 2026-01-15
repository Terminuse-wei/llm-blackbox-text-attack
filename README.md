Environment Setup

1. Create virtual environment (recommended)

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

2. Install dependencies

pip install torch transformers datasets pandas numpy requests tqdm matplotlib scikit-learn

3. Start local LLM with Ollama

Install Ollama, then:

ollama pull llama3:8b
ollama serve

By default, the code uses:

OLLAMA_URL=http://localhost:11434  
OLLAMA_MODEL=llama3:8b

You may override them using environment variables.

⸻

Project Structure

.
├── attacks/                  # Attack logic & utilities
├── prompts/                  # Prompt templates for TextFooler / PWWS / DeepWordBug
│   ├── textfooler.txt
│   ├── pwws.txt
│   └── deepwordbug.txt
├── data/                     # Cached dataset files
├── results/                  # Raw logs + summary + plots
├── run_llm_attack_once.py    # Run attack on one sample
├── run_llm_attack_loop.py    # Batch evaluation
├── summarize_llm_results.py  # Aggregate metrics
├── plot_results.py           # Visualization
└── README.md


⸻

Running Experiments

Run one attack (single example)

python run_llm_attack_once.py

This script loads SST-2 validation data, queries the victim classifier, generates adversarial candidates via LLM prompting, evaluates prediction changes, and saves results into results/.

⸻

Run full evaluation loop

python run_llm_attack_loop.py

The script evaluates three attack styles: TextFooler, PWWS, and DeepWordBug.
All raw outputs are saved as JSON files under results/.

⸻

Summarize results

python summarize_llm_results.py

Produces:

results/comparison.csv

which contains aggregated success rates and statistics.

⸻

Plot performance

python plot_results.py

Produces:

results/success_rate.png


⸻

Notes
	•	Victim model: DistilBERT fine-tuned on SST-2
	•	Dataset: SST-2 validation split
	•	All attacks are black-box and rely only on model predictions.

⸻

Reproducibility

Ensure that Ollama is running before executing the scripts.
Except for LLM generation randomness, all experiments are deterministic.

⸻
