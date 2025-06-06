# wer_eval_domains.py
from jiwer import wer
from collections import defaultdict
import os

MODES = ["char", "sudachi_A", "sudachi_B", "sudachi_C"]
REF_PATH = "/mnt/data/asr_ground_truth_with_domain.txt"
HYP_DIR = "/mnt/data/asr_lm_outputs_real"

# Load references with domains
references = {}
domains = {}
with open(REF_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            uid, ref_text, domain = parts
            references[uid] = ref_text
            domains[uid] = domain

domain_set = set(domains.values())

print("Domain-separated WER evaluation:\n")

for mode in MODES:
    hyp_path = os.path.join(HYP_DIR, f"output_{mode}.txt")
    hyp_lines = {}
    with open(hyp_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                uid, hyp_text = parts
                hyp_lines[uid] = hyp_text

    # Prepare refs and hyps per domain
    refs_by_domain = defaultdict(list)
    hyps_by_domain = defaultdict(list)

    for uid in references:
        if uid in hyp_lines:
            domain = domains.get(uid, "unknown")
            refs_by_domain[domain].append(references[uid])
            hyps_by_domain[domain].append(hyp_lines[uid])

    print(f"Mode: {mode}")
    for domain in sorted(domain_set):
        if len(refs_by_domain[domain]) == 0:
            print(f"  - {domain}: No data")
            continue
        error = wer(refs_by_domain[domain], hyps_by_domain[domain])
        print(f"  - {domain}: WER = {error:.4f}")
    print()

