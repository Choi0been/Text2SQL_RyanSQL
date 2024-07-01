import json
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import model_ko as Model

#class SimpleConfig:
#    def __init__(self, lang='ko'):
#        self.lang = lang  # 설정 언어 기본값 설정
#
#    def __setattr__(self, name, value):
#        super().__setattr__(name, value)

def add_config(config):
    with open("config.json", "r") as f:
        config_file = json.load(f)
    for k, v in config_file[config.lang].items():
        setattr(config, k, v)

def load_model(config):
    model = Model.Model(config)  # config 객체를 생성자에 전달
    model_path = './save_model/best_ckpt/best_ckpt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main(config):
    #config = SimpleConfig()  # config 객체 생성
    add_config(config)  # JSON에서 설정 로드

    # 시드 및 기타 설정
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    if "large" in config.pretrained_model :
        config.hidden_size = 1024
    else :
        config.hidden_size = 768
        
    model = load_model(config)  # 모델 로드
    
    print("자연어 질의를 입력하세요:")
    query = input()  # 사용자로부터 자연어 질의 입력 받기

    sql_prediction = model.predict(query)  # 입력받은 질의를 SQL로 변환
    print("예측된 SQL 문:")
    print(sql_prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--random_seed", default=42, type=int, help="random state (seed)")
    parser.add_argument("--epoch", default=10000, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--max_seq", default=512, type=int, help="maximum length of long text sequence i.e. pretrain segment")

    parser.add_argument("--optimizer_epsilon", default=1e-8, type=float, help="epsilon value for optimzier")
    parser.add_argument("--weight_decay", default=0.9, type=float, help="L2 weight decay for optimizer")
    parser.add_argument("--dropout", default=0.1, type=float, help="probability of dropout layer")

    parser.add_argument("--lang", help="select language\n", choices=["ko", "en"], default="ko")
    parser.add_argument("--train", action="store_true", help="scatter paragraph into paragraphs")
    parser.add_argument("--save_ckpt", default="save_model/test", type=str)
    parser.add_argument("--ckpt_name", default="best_ckpt", type=str)
    parser.add_argument("--device", default=2, type=int)
    args = parser.parse_args()

    main(args)
