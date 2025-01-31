import os
from typing import Optional

import torch
from flask import Flask, jsonify, request
from model import LlamaForCausalLM
from transformers import AutoTokenizer

app = Flask(__name__)


class ModelConfig:
    def __init__(self):
        self.vocab_size = 49152
        self.hidden_size = 576
        self.intermediate_size = 1536
        self.num_hidden_layers = 30
        self.num_attention_heads = 9
        self.num_key_value_heads = 3
        self.hidden_act = "silu"
        self.max_position_embeddings = 512
        self.initializer_range = 0.041666666666666664
        self.rms_norm_eps = 1e-5
        self.tie_word_embeddings = True
        self.pad_token_id = None
        self.bos_token_id = 0
        self.eos_token_id = 0


def load_model(checkpoint_path):
    """Load the model and tokenizer."""
    # Initialize model config
    model_config = ModelConfig()  # Use the class instead of dictionary

    # Initialize model
    model = LlamaForCausalLM(model_config)

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

    # Move model to available device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if torch.backends.mps.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )
    model = model.to(device)
    model.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, max_length=100):

    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        generated_tokens = input_ids[0].tolist()

        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[..., -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)

            generated_tokens.append(next_token.item())
            next_token = next_token.unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# Load model globally
print("Loading model and tokenizer...")
checkpoint_path = "checkpoint_5050.pt"
model, tokenizer, device = load_model(checkpoint_path)
print(f"Model loaded and running on {device}")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data["prompt"]
        max_length = data.get("max_length", 100)

        generated_text = generate_text(model, tokenizer, device, prompt, max_length)
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
