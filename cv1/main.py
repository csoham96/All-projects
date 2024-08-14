from paddleocr import PaddleOCR
import config
import logging
import transformers
import bitsandbytes as bnb
import os
from fastapi import FastAPI, File, UploadFile
from doc_summarize import doc_summarize

app = FastAPI() 
summarizer =doc_summarize()
@app.post("/summarize/")
async def summarize_image(image: UploadFile = File(...)):
    # Use the summarize_image method from the ImageSummarizer class
    summary = summarizer.summarize_image(image)
    print(summary)
    return {"summary": summary}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)