# EDA 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2 
import optparse
from db_config import get_db_engine

# 결과 저장 폴더 생성
RESULT_DIR = 'analsys_results'
os.makedirs(RESULT_DIR, exist_ok=True)

def load_data_from_db():
    engine = get_db_engine()  # 필요한 데이터 가져오기

    # 테이블 목록 
    tables = {
        'diseases': 'SELECT * FROM diseases',
        'hospital_types' :'SELECT * FROM hospital_types',
        'regions' : 'SELECT * FROM regions',
        'age_based_costs' : 'SELECT * FROM age_based_costs',
        'hospital_based_costs' : 'SELECT * FROM hospital_based_costs',
        'region_based_costs' : 'SELECT * FROM region_based_costs',
    }

    data = {}
    for key , query in tables.items():
        try:
            data[key] = pd.read_sql_query(query, engine)
            print(f"{key} 테이블에서 {len(data[key])} 행을 불러왔습니다.")
        except Exception as e:
            print(f"{key} 불러오기 실패")
            data[key] = pd.DataFrame()  # 빈 프레임 생성
    return data

# 다빈도 질병 
def anaylsis_disease_costs(data):
    if 'diseases' not in data or data['diseases'].empty:
        print("질병 데이터가 없습니다.")
        return
    
    disease_cost = data['diseases']