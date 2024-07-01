import torch
import tokenization
from model_ko import Model
import data_ko as Data
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="save_model/best_ckpt/best_ckpt",
                        help="Path to the model checkpoint")
    parser.add_argument("--device", type=int, default=0, help="Device to run the model on")
    parser.add_argument("--pretrained_model", type=str, default="monologg/koelectra-base-v3-discriminator")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")  # dropout 인자 추가

    args = parser.parse_args()

    # Configure device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    config = {
        'pretrained_model': args.pretrained_model,
        'device': args.device,
        'hidden_size': 768,  # 예시 값, 필요에 따라 조정하세요
        'max_num': {  # 예시 값, 실제 모델이 필요로 하는 값으로 채워야 합니다.
            'table_id' : 2, 
            'table_num': 2,  # 테이블 수
            'column_num': 48,  # 컬럼 수
            'select': 5,
            'groupby': 3,
            'orderby': 2,
            'where': 6,
            'having': 2,
            'spc_id' : 2
            },
        'dropout': args.dropout
    }
    tokenizer = tokenization.get_tokenizer(config['pretrained_model'])
    model = Model.load(args.ckpt_path, config).to(device)
    model.eval()

    # User input
    user_input = input("Enter your natural language query: ")
    tokens, token_ids = tokenization.tokenize(user_input, tokenizer)

    # Prepare data as expected by the model
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([[1] * len(token_ids)], dtype=torch.long).to(device)
    q_len = [len(token_ids)]  # Assuming the entire input is one query
    c_len = [[1]]  # No columns explicitly provided
    t_len = [[1]]  # No tables explicitly provided
    sql_mask = [1] * len(token_ids)  # Assuming entire input is relevant to SQL

    # Predict SQL
    with torch.no_grad():
        sql_query = model.generate_sql(input_ids, attention_mask,q_len,c_len,t_len,sql_mask)

    print("Predicted SQL Query:", sql_query)

if __name__ == "__main__":
    main()

