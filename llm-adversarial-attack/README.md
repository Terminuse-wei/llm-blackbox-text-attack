# LLM-based Adversarial Attacks (PoC)

## Goal
This project studies whether large language models can emulate classical NLP adversarial attacks purely through prompt engineering, without access to gradients, saliency maps, or internal model states.

## Threat Model
We consider black-box attacks on a sentiment classifier (DistilBERT SST-2). The attacker only observes input text and model predictions.

## Attack Recipes Emulated
- **TextFooler** (Jin et al., EMNLP 2019)
- **PWWS** (Ren et al., ACL 2019)
- **DeepWordBug** (Gao et al., ACL 2018)

Each recipe is translated into a natural-language prompt that encodes its core constraints and objectives.

## Architecture
LLM (Llama 3 via Ollama) → Prompt-based attack → Victim model → Evaluation

## Key Findings
LLM successfully reproduces the functional behavior of TextFooler-style attacks with ~50% success rate on SST-2 validation samples. Character-level attacks (DeepWordBug) and saliency-style attacks (PWWS) were significantly weaker when implemented purely via prompting.

## Ethical Use
This work is intended for security research and robustness evaluation only.
