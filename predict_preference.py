import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
import argparse
import pandas as pd

def load_model(model_path: str, checkpoint_path: str, device: str):
    print(f"Loading base model and tokenizer from {model_path}...")

    if "cuda" in device:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    else:
        kwargs = {"torch_dtype": torch.float32}

    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        **kwargs,
    )

    print(f"Loading fine-tuned weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if the checkpoint is a state_dict or a dict containing the state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict, strict=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    
    print("Model and tokenizer with fine-tuned weights loaded successfully.")
    return model, tokenizer

def predict_user_preference(
    model, 
    tokenizer, 
    liked_movies: list, 
    target_item_title: str,
    device: str
):
    item_title_list = ", ".join([f'"{movie}"' for movie in liked_movies if movie])

    prompt = f"""#Question: A user has given high ratings to the following movies: {item_title_list}. Based on this information, would the user enjoy the movie titled {target_item_title}? Please answer "Yes" or "No", and explain the reason for your answer. \n#Answer:"""

    print("\n--- Generating with the following prompt ---")
    print(prompt)
    print("------------------------------------------")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    response_ids = output_ids[0][len(inputs['input_ids'][0]):]
    prediction = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/minseong/CoLLM/Vicuna/Vicuna-7B-v1.3",
        help="Path to the BASE model directory (e.g., Vicuna-7B).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/minseong/CoLLM/pth-movie/stage_2_ours/checkpoint_best.pth",
        help="Path to the fine-tuned .pth checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to run the model on.",
    )
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_path, args.checkpoint_path, args.device)

    df = pd.read_pickle("/home/minseong/CoLLM/collm-datasets/ml-1m/test_ood2_genres_normalized.pkl")

    for idx in range(100):
        test_ = df[idx:idx+1]
        liked_movies        = test_['his_title'].values[0]
        target_item_title   = test_['title'].values[0]
        ground_truth        = test_['label'].values[0]

        prediction = predict_user_preference(
            model,
            tokenizer,
            liked_movies,
            target_item_title,
            args.device
        )

        print("========================================")
        print(f"[Index {idx}] ✅ Model Answer: {prediction}")
        print(f"[Index {idx}] 🎯 Ground Truth: {ground_truth}")
        print("========================================\n")