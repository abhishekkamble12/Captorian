from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


def bleu_scores(references, hypotheses):
    smoothing = SmoothingFunction().method1
    return {
        'BLEU-1': corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        'BLEU-2': corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing),
        'BLEU-3': corpus_bleu(references, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smoothing),
        'BLEU-4': corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing),
    }
