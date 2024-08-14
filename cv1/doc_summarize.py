from paddleocr import PaddleOCR
import config
import logging
import transformers
import os
from fastapi import UploadFile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class load_models():
    """
    Class to handle loading and managing OCR and LLM models.
    """
    def __init__(self) -> None:
        """
        Initialize the class with configuration values.
        """
        self.use_gpu=config.use_gpu
        self.device_map=config.device_map
        self.REPO_ID=config.REPO_ID
    def load_ocr(self):
        """
        Loads the PaddleOCR model for text extraction.
        """
        ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=self.use_gpu)
        return ocr
    def load_llm(self):
        """
        Loads the Llama LLM model for text summarization.
        """
        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path = self.REPO_ID, 
            device_map=config.device_map, 
            offload_folder="/tmp/.offload",
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            )
        return model

class doc_summarize():
    """
    Class to handle text extraction, summarization, and API endpoint.
    """
    def __init__(self) -> None:
        """
        Initialize the class with configuration and model loading.
        """
        self.REPO_ID=config.REPO_ID
        self.device_map=config.device_map
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(config.REPO_ID)

        lm=load_models()
        logging.info("Loading Ocr Model")
        self.ocr=lm.load_ocr()
        logging.info("Loading LLM ")
        self.llm=lm.load_llm()
    def extract_text(self,image_path):
        """
        Extracts text from an image using the PaddleOCR model.
        """
        Result = self.ocr.ocr(image_path, cls=True)
        result = Result[0]
        txts = [line[1][0] for line in result]
        all_words=" ".join(txts)
        if len(all_words)==0:
            raise Exception("No words present so cant be summarized,try uploading an image with some words on it")
        return all_words
    def summarize_text(self,text):
        """
        Summarizes a given text using the Llama LLM model.
        """
        batch = self.tokenizer(
            f"Summarize this text: {text}",
            return_tensors="pt", 
            add_special_tokens=False
        )
        batch = {k: v for k, v in batch.items()}
        n_input_tokens = batch["input_ids"].shape[-1]
        generated = self.llm.generate(batch["input_ids"].to("cuda"), max_length=n_input_tokens+25)
        summary=self.tokenizer.decode(generated[0],skip_special_tokens=True)
        return summary

    def summarize_image(self,image: UploadFile):
        """
        Performs image upload, text extraction, and summarization.
        """
        contents = image.file.read()
        image_path = "temp_image.png"
        
        with open(image_path, "wb") as f:
            f.write(contents)
        logging.info("Starting to Extract text from image")
        text=self.extract_text(image_path)
        logging.info("Starting to summarize")
        summary = self.summarize_text(text)
        logging.info("Finished sumarizing")
        return  summary
