import config
from jiwer import wer, cer
from doc_summarize import doc_summarize
import requests
from rouge_score import rouge_scorer
import time
API_URL = config.API_URL

def evaluate_image(image_path):
    """
    Evaluates the extracted text and summary of an image, measuring API latency.

    Args:
        image_path: Path to the image file.

    Returns:
        (evaluation_metrics): Tuple containing
        evaluation metrics and API latency.
    """
    start_time = time.time()

    # Use predicted_text function from your API (assuming it returns text)
    try:
        response = requests.post(API_URL, files={"image": open(image_path, "rb")})
        response.raise_for_status()  # Raise exception for non-200 status codes
        data = response.json()
        generated_summary = data["summary"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        predicted_text = None  # Handle potential errors gracefully

    end_time = time.time()
    latency = end_time - start_time    
    # Implement your evaluation logic here (e.g., using rouge metrics)
    reference_summary=config.reference_summary #configure in config.py
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    ROUGE1=scores['rouge1'].fmeasure
    ROUGEL=scores['rougeL'].fmeasure
    evaluation_metrics = (latency,ROUGE1,ROUGEL)

    return evaluation_metrics

all_latency=[]
all_metrics=[]
# Iterate through your image data
for image_path in config.your_image_data:
  image_summarizer=doc_summarize()
  evaluation_metrics = evaluate_image(image_path,image_summarizer)
  latency=evaluation_metrics[0]
  metrics=evaluation_metrics[1],evaluation_metrics[2]
  all_latency.append(latency)
  all_metrics.append(metrics)

print("Average latency:",sum(all_latency)/len(all_latency))
print("Evaluation for rouge metrics:",all_metrics)