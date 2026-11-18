"""Shared local MMQA set metric; not the official benchmark scorer."""
import json
from src.utils import compute_exact_match, compute_f1

def score_answer(prediction, gold_answers, metadata):
    """MMQA gold entries are jointly required answers, not alternative aliases."""
    if metadata.get("dataset") == "multimodalqa" and len(gold_answers) > 1:
        from scipy.optimize import linear_sum_assignment
        if isinstance(prediction, list):
            predicted = prediction
        else:
            try:
                predicted = json.loads(str(prediction))
            except (ValueError, TypeError):
                predicted = str(prediction).split(";")
            if not isinstance(predicted, list):
                predicted = [str(prediction)]
        predicted = list(dict.fromkeys(str(v).strip() for v in predicted if str(v).strip()))
        gold_answers = list(dict.fromkeys(map(str, gold_answers)))
        if not predicted:
            return 0, 0.0, 0.0, 0.0
        scores = [[compute_f1(p, g)[0] for g in gold_answers] for p in predicted]
        rows, cols = linear_sum_assignment(scores, maximize=True)
        overlap = sum(scores[i][j] for i, j in zip(rows, cols))
        precision, recall = overlap / len(predicted), overlap / len(gold_answers)
        em = int(len(predicted) == len(gold_answers) and all(
            compute_exact_match(predicted[i], gold_answers[j]) for i, j in zip(rows, cols)))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return em, f1, precision, recall
    scores = [(compute_exact_match(prediction, g), *compute_f1(prediction, g)) for g in gold_answers]
    if not scores:
        return 0, 0.0, 0.0, 0.0
    best = max(scores, key=lambda x: x[1])
    return max(x[0] for x in scores), *best[1:]
