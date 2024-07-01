import torch
import tokenization
from model_ko import Model
import data_ko as Data
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="save_model/best_ckpt",
                        help="Path to the model checkpoint")
    parser.add_argument("--device", type=int, default=0, help="Device to run the model on")
    parser.add_argument("--pretrained_model", type=str, default="monologg/koelectra-base-v3-discriminator")
    args = parser.parse_args()

    # Configure device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    config = {
        'ckpt_path': args.ckpt_path,
        'device': args.device,
        'pretrained_model': args.pretrained_model
    }
    tokenizer = tokenization.get_tokenizer(config['pretrained_model'])
   
    model = Model.load(args.ckpt_path, {
    'pretrained_model': args.pretrained_model,
    'device': args.device
    }).to(device)
    
    model.eval()

    # User input
    user_input = input("Enter your natural language query: ")
    tokens, token_ids = tokenization.tokenize(user_input, tokenizer)

    # Prepare data as expected by the model
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([[1] * len(token_ids)], dtype=torch.long).to(device)

    # Predict SQL
    with torch.no_grad():
        sql_query = model.generate_sql(input_ids, attention_mask)

    print("Predicted SQL Query:", sql_query)

if __name__ == "__main__":
    main()

