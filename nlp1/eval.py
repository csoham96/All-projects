from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Cosine Similarity
def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# BLEU Score
def compute_bleu_score(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)

# ROUGE Score
def compute_rouge_score(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]

# Example usage
embedding1 = np.array([0.1, 0.2, 0.3])
embedding2 = np.array([0.1, 0.2, 0.3])
similarity = compute_cosine_similarity(embedding1, embedding2)
print("Cosine Similarity:", similarity)

reference_text = "The quick brown fox jumps over the lazy dog."
candidate_text = "The fast brown fox leaps over the lazy dog."
bleu = compute_bleu_score(reference_text, candidate_text)
print("BLEU Score:", bleu)

rouge_scores = compute_rouge_score(reference_text, candidate_text)
print("ROUGE Scores:", rouge_scores)
