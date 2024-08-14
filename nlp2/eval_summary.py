from rouge_score import rouge_scorer

def calculate_rouge_score(reference_summary, generated_summary):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    scores = scorer.score(reference_summary, generated_summary)
    
    return scores

# Example usage
reference_summary = "This is the correct summary of the document."
generated_summary = "This is the summary of the document."

rouge_scores = calculate_rouge_score(reference_summary, generated_summary)
print("ROUGE Scores:", rouge_scores)
