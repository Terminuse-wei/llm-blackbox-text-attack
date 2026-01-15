# Attack Specifications

## TextFooler (Jin et al., EMNLP 2019)

Type: Word-level adversarial attack

Key properties:
- Replace important words
- Preserve semantic meaning
- Minimal number of changes
- Flip the model prediction

Prompt constraints:
- Prefer synonyms or near-synonyms
- Do not modify named entities
- Avoid changing sentiment polarity
- Keep changes minimal

---

## PWWS (Ren et al., ACL 2019)

Type: Saliency-based word substitution attack

Key properties:
- Identify important words
- Replace important words with synonyms
- Preserve semantics
- Flip the model prediction

Prompt constraints:
- Modify only a few highly influential words
- Preserve original meaning
- Avoid grammatical errors

---

## DeepWordBug (Gao et al., ACL 2018)

Type: Character-level adversarial attack

Key properties:
- Introduce small character-level perturbations
- Maintain human readability
- Minimal visual change
- Flip the model prediction

Prompt constraints:
- Use character insertion, deletion, or swap
- Do not change the full word
- Keep the sentence readable