from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pathlib import Path
import os

# Adjust the path to the model directory inside the 'models' folder
model_dir = Path("models/bert_model")

# Verify directory structure first
config_path = model_dir / "config.json"
model_path = model_dir / "model.safetensors"  # Use model.safetensors instead of pytorch_model.bin
vocab_path = model_dir / "vocab.txt"

# Check if all required files are present
if not all([config_path.exists(), model_path.exists(), vocab_path.exists()]):
    raise FileNotFoundError(f"Missing model files in {model_dir}. Required files: config.json, model.safetensors, vocab.txt")

# Try loading the tokenizer and model with error handling
try:
    # Load the tokenizer and model with explicit local_files_only flag and better error handling
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir.as_posix(),  # Convert path to POSIX format
        local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir.as_posix(),  # Convert path to POSIX format
        local_files_only=True
    )
except Exception as e:
    print(f"Loading failed. Potential causes: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}") from e

# Define the function to detect bias in a given text
def detect_bias(text):
    try:
        # Tokenize the input text
        inputs = tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            add_special_tokens=True
        )
        
        # Perform inference with the model
        with torch.inference_mode():  # Better than torch.no_grad()
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze()
        
        # Get the label mapping from the model's config
        labels = {int(k): v for k, v in model.config.id2label.items()}
        
        # Get the predicted label and its confidence score
        max_idx = torch.argmax(probs).item()
        
        # Return the result with probabilities for each label
        return {
            "label": labels[max_idx],
            "score": probs[max_idx].item(),
            "probabilities": {
                label: probs[idx].item() for idx, label in labels.items()
            }
        }
        
    except Exception as e:
        print(f"Inference error with text: {text[:50]}...")
        raise RuntimeError(f"Inference failed: {str(e)}") from e

# Example usage
if __name__ == "__main__":
    text = "This is an example news article that might contain biased content."
    try:
        result = detect_bias(text)
        print("Analysis Result:")
        print(f"Predicted Label: {result['label']}")
        print(f"Confidence: {result['score']:.4f}")
        for label, prob in result['probabilities'].items():
            print(f"{label}: {prob:.4f}")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
