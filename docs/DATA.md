# Khwarizmi AI — Data Strategy

> **Quality > quantity.** Arabic + Egyptian Arabic + English + code + math + reasoning +
> planning + tool-use + verification. Offline, contamination-controlled.

## 1. Domains & sources (offline-friendly)
- **English text/code:** permissively licensed corpora, public code (compatible licenses).
- **Arabic / Egyptian Arabic:** curated Arabic text; Egyptian dialect where useful; **no
  scraping of copyrighted or private data**.
- **Coding:** Python-heavy; verified by the Python Brain / Safe Exec.
- **Mathematics & reasoning:** problem→solution pairs; synthetic where high-quality.
- **Planning / project mgmt:** synthesized goal→plan graphs (with the Project Planner).
- **Tool use & verification:** traces where a local tool improved the outcome.

## 2. Pipeline stages
1. **Ingest** (licensed only) → 2. **Clean** (normalize, dedup) → 3. **Filter** (quality,
   toxicity, PII) → 4. **Deduplicate** (near-dup by embedding/hash) → 5. **Contamination
   check** (against eval sets) → 6. **Tokenize** (trained BPE/Unigram) → 7. **Version &
   manifest** → 8. **Audit log** (offline).

## 3. Contamination & leakage controls
- Exact + fuzzy overlap detection between train and benchmark sets; remove overlaps.
- No benchmark prompts in training; no solution leakage for reasoning/coding evals.
- Versioned dataset hashes; reproducible manifests.

## 4. Synthetic data discipline
- Generate with offline tooling; **verify** every synthetic reasoning/coding example with
  local deterministic checks before admission.
- Avoid low-quality template spam; cap repetition; human-spot-check samples.

## 5. Multilingual balance
- Explicit EN/AR/Egyptian/Franco/code mixing ratios; document the mix; ensure the tokenizer
  covers all scripts (Latin + Arabic + code symbols).

## 6. Storage
- Local, versioned, checksummed. No external dependency at training or inference time.
