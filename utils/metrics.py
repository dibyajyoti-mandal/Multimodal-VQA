import math
from collections import Counter


def ngram_precision(reference, candidate, n):
    ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
    cand_ngrams = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])

    overlap = sum(min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0)) for ngram in cand_ngrams)
    total = sum(cand_ngrams.values())

    return overlap / total if total > 0 else 0


def brevity_penalty(reference, candidate):
    ref_len = len(reference)
    cand_len = len(candidate)

    if cand_len > ref_len:
        return 1
    else:
        return math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0


def compute_bleu(reference_sentences, candidate_sentences, max_n=4):
    assert len(reference_sentences) == len(candidate_sentences), "reference and candidate counts must match"

    bleu_scores = []
    for ref, cand in zip(reference_sentences, candidate_sentences):
        precisions = [ngram_precision(ref, cand, n) for n in range(1, max_n+1)]
        geometric_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n) if any(precisions) else 0
        bp = brevity_penalty(ref, cand)
        bleu_scores.append(bp * geometric_mean)

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
