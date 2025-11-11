# compare_llms.py
import time
import json
import os
from pathlib import Path
from typing import List, Dict
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- Настройки ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "results.json"

# Модели для тестирования — подбери нужные
MODELS = [
    {"name": "gpt2", "type": "causal"},
    {"name": "distilgpt2", "type": "causal"},
    {"name": "cointegrated/rut5-base-multitask", "type": "seq2seq"},
]

# Параметры генерации
GEN_KW = dict(max_new_tokens=64, do_sample=False, temperature=0.7, top_p=0.9)


# --- Утилиты для чтения тестов ---
def read_lines(p: Path) -> List[str]:
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


# --- Перплексити (для causal models) ---
def compute_perplexity_causal(model, tokenizer, texts: List[str], device):
    model.eval()
    ppls = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=1024).to(
            device
        )
        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss.item()
            ppl = math.exp(loss) if loss < 100 else float("inf")
            ppls.append(ppl)
    return sum(ppls) / len(ppls) if ppls else None


# --- Perplexity для seq2seq (approx) ---
def compute_perplexity_seq2seq(model, tokenizer, texts: List[str], device):
    model.eval()
    ppls = []
    for t in texts:
        # encode input and labels = same as target (approx)
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(
            device
        )
        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss.item()
            ppl = math.exp(loss) if loss < 100 else float("inf")
            ppls.append(ppl)
    return sum(ppls) / len(ppls) if ppls else None


# --- Coherence (BLEU of generated continuation vs ref) ---
def coherence_score(generator_fn, test_pairs, tokenizer, device):
    smoothie = SmoothingFunction().method4
    scores = []
    for pair in test_pairs:
        prompt = pair["prompt"]
        ref = pair["ref"]
        gen = generator_fn(prompt)
        # simple tokenization by whitespace
        ref_tokens = [ref.split()]
        hyp_tokens = gen.split()
        try:
            sc = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
        except Exception:
            sc = 0.0
        scores.append(sc)
    return sum(scores) / len(scores) if scores else None


# --- Factual QA: simple matching / normalized compare ---
def normalize_text(s: str):
    return "".join(c.lower() for c in s if c.isalnum() or c.isspace()).strip()


def factual_score(generator_fn, qa_pairs):
    correct = 0
    total = 0
    for qa in qa_pairs:
        q = qa["q"]
        a = qa["a"]
        gen = generator_fn(q)
        gen_norm = normalize_text(gen).strip()
        a_norm = normalize_text(a).strip()
        # simple substring or exact compare:
        if a_norm and (a_norm in gen_norm or gen_norm in a_norm):
            correct += 1
        total += 1
    return correct, total, (correct / total if total else None)


# --- Main runner for one model ---
def evaluate_model(mdef, perp_texts, cont_pairs, qa_pairs, device):
    name = mdef["name"]
    mtype = mdef.get("type", "causal")
    print(f"\n=== Evaluating {name} ({mtype}) ===")
    result = {"model": name, "type": mtype}

    # load model & tokenizer
    device_idx = 0 if torch.cuda.is_available() and device == "cuda" else -1
    if mtype == "causal":
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).to(device)

        # generator function
        def gen_fn(prompt):
            inputs = tok(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, **GEN_KW)
            return tok.decode(out[0], skip_special_tokens=True)[
                len(tok.decode(inputs["input_ids"][0], skip_special_tokens=True)) :
            ].strip()

        # perplexity
        ppl = compute_perplexity_causal(model, tok, perp_texts, device)
    else:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)

        def gen_fn(prompt):
            inputs = tok(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, **GEN_KW)
            return tok.decode(out[0], skip_special_tokens=True).strip()

        ppl = compute_perplexity_seq2seq(model, tok, perp_texts, device)

    result["perplexity"] = ppl
    # time measurement for generation average
    times = []
    gens = []
    for prompt in (pair["prompt"] for pair in cont_pairs[:10]):
        t0 = time.time()
        _ = gen_fn(prompt)
        t1 = time.time()
        times.append(t1 - t0)
    result["avg_gen_time_s"] = sum(times) / len(times) if times else None

    # coherence
    coh = coherence_score(gen_fn, cont_pairs[:50], tok, device)
    result["coherence_bleu"] = coh

    # factual QA
    correct, total, acc = factual_score(gen_fn, qa_pairs)
    result["factual_correct"] = correct
    result["factual_total"] = total
    result["factual_acc"] = acc

    # sample generations (first 3)
    samples = []
    for pair in cont_pairs[:3]:
        samples.append(
            {
                "prompt": pair["prompt"],
                "ref": pair["ref"],
                "gen": gen_fn(pair["prompt"]),
            }
        )
    result["samples"] = samples

    return result


def main():
    nltk.download("punkt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perp_texts = read_lines(DATA_DIR / "perp_texts.txt")
    cont_pairs = read_jsonl(DATA_DIR / "continuations.jsonl")
    qa_pairs = read_jsonl(DATA_DIR / "factual_qa.jsonl")
    results = []
    for m in MODELS:
        try:
            res = evaluate_model(m, perp_texts, cont_pairs, qa_pairs, device)
            results.append(res)
            # save intermediate
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed evaluating {m['name']}: {e}")
    print("\n== All done. Results saved to", OUT)


if __name__ == "__main__":
    main()
