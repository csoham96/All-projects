import nltk
from nltk.translate.bleu_score import sentence_bleu

# Download necessary NLTK data
nltk.download('punkt')

def calculate_bleu_score(reference_translation, generated_translation):
    # Tokenize the sentences
    reference = [nltk.word_tokenize(reference_translation)]
    generated = nltk.word_tokenize(generated_translation)
    
    # Calculate BLEU score
    bleu_score = sentence_bleu(reference, generated)
    
    return bleu_score

# Example usage
reference_translation = "This is the correct translation."
generated_translation = "This is the translation."

bleu_score = calculate_bleu_score(reference_translation, generated_translation)
print("BLEU Score:", bleu_score)
