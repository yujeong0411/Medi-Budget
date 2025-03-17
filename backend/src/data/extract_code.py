import pandas as pd
from db_config import get_db_engine, insert_dataframe

def extract_disease_codes():
    try: 
        engine = get_db_engine()
        
        # SELECT DISTINCT : 중복제거
        query = """
        SELECT DISTINCT disease_code, disease_name
        FROM raw_medical_data
        WHERE disease_code IS NOT NULL
        ORDER BY disease_code
        """

        # 쿼리 실행 및 DataFrame으로 변환
        disease_codes_df = pd.read_sql(query, engine)

        print(f"총 {len(disease_codes_df)}개의 고유 질병 코드가 추출되었습니다.")
        return disease_codes_df
    
    except Exception as e:
        print(f"질병 코드 추출 에러: {e}")
        return pd.DataFrame()
    

if __name__ == '__main__':
    disease_codes = extract_disease_codes()

    if not disease_codes.empty:
        print(disease_codes.head())

        disease_codes.to_csv("disease_codes.csv", index=False, encoding='utf-8-sig')
        print("질병코드가 CSV 파일로 저장")
        
        # diseases 테이블이 이미 있다면 기존 데이터를 삭제하고 새로 저장 ('replace')
        # 기존 데이터에 추가하려면 'append'로 변경
        try: 
            insert_dataframe(disease_codes, table_name='diseases', if_exists='append')
            print("질병코드가 diseases 테이블에 저장되었습니다.")
        except Exception as e:
            print(f"db저장 실패: {e}")