from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")


def get_db_engine() :
    # 데이터 베이스 엔진을 반환
    connection_string = f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    return create_engine(connection_string)

def execute_query(query, params=None):
    # SQL 쿼리 실행
    engine = get_db_engine()
    with engine.begin() as connection:
        if params:
            result = connection.execute(query, params)
        else:
            result = connection.execute(query)
        return result
    
# def read_sql_query(query, params=None):
#     # SQL 쿼리를 실행하고 결과를 DataFrame으로 반환
#     engine = get_db_engine()
#     return pd.read_sql_query(query, engine, params=params)

def insert_dataframe(df, table_name, if_exists='append'):
    # DataFrame을 데이터베이스 테이블에 저장
    engine = get_db_engine()
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"{len(df)}개의 행이 {table_name} 테이블에 저장")

def test_connection():
    # 데이터베이스 연결 테스트
    try:
        engine = get_db_engine()
        result = pd.read_sql_query("SELECT 1 as test", engine)
        print("데이터베이스 연결 성공")
        print(result)
        return True
    except Exception as e:
        print(f'데이터베이스 연결 실패: {e}')
        return False
    
if __name__ == '__main__':
    test_connection()