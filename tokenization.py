import transformers
# from transformers import BertTokenizer
# from kobert.tokenization_kobert import KoBertTokenizer
# from kobert import get_pytorch_kobert_model

# 240507 test
#def tokenize(text, tokenizer):
#    tokens = tokenizer.tokenize(text)
#    token_ids = tokenizer.convert_tokens_to_ids(tokens)
#    return tokens, token_ids

def get_tokenizer(config) :
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.pretrained_model
        , cache_dir="cache"
    )
    # 240508 test 
    #def get_tokenizer(pretrained_model_name) :
    
    # 240508 main run epoch test를 위한 주석 (original은 아님. config.pretrained_model)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name, cache_dir="cache")
    
    # if config.pretrained_model == "monologg/kobert":
    #     # print("\n\n=== KoBertTokenizer ===\n\n")
    #     # tokenizer = KoBertTokenizer.from_pretrained(
    #     #     config.pretrained_model
    #     #     , cache_dir="cache"
    #     #     )
    #     _, tokenizer = get_pytorch_kobert_model()
    #
    #     # tokenizer.cls_id = 2
    #     # tokenizer.sep_id = 3
    #     # tokenizer.pad_id = 1
    # else:
    #     # print("\n\n=== AutoTokenizer ===\n\n")
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         config.pretrained_model
    #         , cache_dir="cache"
    #         )
    # tokenizer = BertTokenizer.from_pretrained(
    #     config.pretrained_model
    #     , cache_dir="cache"
    #     )

    # tokenizer.cls_id = tokenizer._convert_token_to_id("[CLS]")
    # tokenizer.sep_id = tokenizer._convert_token_to_id("[SEP]")
    # tokenizer.pad_id = tokenizer._convert_token_to_id("[PAD]")
    tokenizer.cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
    tokenizer.sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    tokenizer.pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

    return tokenizer
