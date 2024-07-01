# query_generator.py

# 필요한 모듈과 함수를 import 합니다.
from model_ko import Model  # 모델 로딩을 위한 클래스
from tokenization import tokenize  # 토큰화 함수
from process_sql import process_question  # SQL 처리 함수
from config import model_config  # 모델 구성 설정

class QueryGenerator:
    def __init__(self, model_path):
        self.model = Model.load(model_path)

    def generate_query(self, question):
        # 질문을 토큰화합니다.
        tokens = tokenize(question)
        
        # 토큰화된 질문을 SQL 처리 모듈을 통해 처리합니다.
        processed_question = process_question(tokens)
        
        # 처리된 질문을 모델에 입력하여 SQL 쿼리를 예측합니다.
        sql_query = self.model.predict(processed_question)
        return sql_query

# 모델 경로 설정
model_path = model_config['./save_model/best_ckpt/best_ckpt']

# QueryGenerator 인스턴스 생성
query_gen = QueryGenerator(model_path)

# 사용자의 질문
user_question = "서울시에 있는 약국 중 가양플러스약국의 월요일과 화요일 진료 시간을 알려줘"

# SQL 쿼리 생성
predicted_sql = query_gen.generate_query(user_question)
print(predicted_sql)

