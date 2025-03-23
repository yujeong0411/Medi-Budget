from configs.db_config import get_db_engine, execute_query, insert_dataframe
import pandas as pd
from sqlalchemy import text 

"""pd.read_sql_query를 사용하는 것이 더 나은 경우:
데이터를 조회하여 DataFrame으로 가져올 때
데이터를 바로 분석하거나 처리해야 할 때
코드가 더 간결해집니다

execute_query와 text()를 사용하는 것이 더 나은 경우:
테이블 생성, 삭제, 수정과 같은 DDL 작업
INSERT, UPDATE, DELETE와 같은 데이터 변경 작업
트랜잭션 제어가 필요한 경우"""


# 의료기관 종별 테이블 생성
def create_hospital_types_table():
    engine = get_db_engine()

    # 기존 테이블이 있다면 삭제
    drop_query = text("""
    DROP TABLE IF EXISTS hospital_types;
    """)
    execute_query(drop_query)

    # 테이블 생성
    create_query = text("""
    CREATE TABLE hospital_types (
        hospital_type_id SERIAL PRIMARY KEY,
        hospital_type VARCHAR(100) NOT NULL UNIQUE);
    """)
    execute_query(create_query)

    # 원본 데이터에서 추출
    unique_types_query = text("""
    SELECT DISTINCT hospital_type
    FROM raw_medical_data
    WHERE hospital_type IS NOT NULL
    ORDER BY hospital_type;
    """)

    # 고유한 병원 가져오기
    hospital_types_df = pd.read_sql_query(unique_types_query, engine)
    print(f"고유한 의료기관 종별: {len(hospital_types_df)}개")

    # 데이터 삽입
    insert_dataframe(hospital_types_df, 'hospital_types', if_exists='append')

    return hospital_types_df


# 지역 테이블 생성
def create_regions_table():
    engine = get_db_engine()

    # 기존 테이블 삭제
    drop_query = text("""
    DROP TABLE IF EXISTS regions;
    """)
    execute_query(drop_query)

    # 테이블 생성
    create_query = text("""
    CREATE TABLE regions (
        region_id SERIAL PRIMARY KEY,
        region_name VARCHAR(100) NOT NULL UNIQUE);
    """)
    execute_query(create_query)

    # 원본 데이터 추출
    unique_regions_query = text("""
    SELECT DISTINCT region_name
    FROM raw_medical_data
    WHERE region_name IS NOT NULL
    ORDER BY region_name;
    """)

    # 고유한 지역 가져오기
    regions_df = pd.read_sql_query(unique_regions_query, engine)
    print(f"고유한 지역: {len(regions_df)}개")

    # 데이터 삽입
    insert_dataframe(regions_df, 'regions', if_exists='append')

    return regions_df


# 진료비 테이블 초기화
def create_costs_table():
    engine = get_db_engine()

    # 병원종류별 진료비 테이블 생성
    drop_hospital_costs_query = text("""
    DROP TABLE IF EXISTS hospital_based_costs;
    """)
    execute_query(drop_hospital_costs_query)

    create_hospital_costs_query = text("""
    CREATE TABLE hospital_based_costs (
        hospital_costs_id SERIAL PRIMARY KEY,
        hospital_type_id INT NOT NULL,
        hospital_type VARCHAR(100) NOT NULL,
        disease_id INT NOT NULL,
        disease_name VARCHAR(100) NOT NULL,
        patient_count INT,
        visit_count INT,
        medical_care_cost_total NUMERIC,
        insurance_payment NUMERIC,
        patient_payment NUMERIC,
        patient_payment_per_visit NUMERIC,
        patient_payment_per_patient NUMERIC,
        year INT,
        FOREIGN KEY (hospital_type_id) REFERENCES hospital_types(hospital_type_id),
        FOREIGN KEY (disease_id) REFERENCES diseases(disease_id)
    );
    """)
    execute_query(create_hospital_costs_query)
    print("hospital_based_costs 테이블이 생성되었습니다.")
    
    # 지역별 진료비 테이블 생성
    drop_region_costs_query = text("""
    DROP TABLE IF EXISTS region_based_costs;
    """)
    execute_query(drop_region_costs_query)

    create_region_costs_query = text("""
    CREATE TABLE region_based_costs (
        region_costs_id SERIAL PRIMARY KEY,
        region_id INT NOT NULL,
        region_name VARCHAR(100) NOT NULL,
        disease_id INT NOT NULL,
        disease_name VARCHAR(100) NOT NULL,
        patient_count INT,
        visit_count INT,
        medical_care_cost_total NUMERIC,
        insurance_payment NUMERIC,
        patient_payment NUMERIC,
        patient_payment_per_visit NUMERIC,
        patient_payment_per_patient NUMERIC,
        year INT,
        FOREIGN KEY (region_id) REFERENCES regions(region_id),
        FOREIGN KEY (disease_id) REFERENCES diseases(disease_id)
        );
    """)
    execute_query(create_region_costs_query)
    print("region_based_costs 테이블이 생성되었습니다.")

    # 3. age_based_costs 테이블 생성
    drop_age_costs_query = text("""
    DROP TABLE IF EXISTS age_based_costs;
    """)
    execute_query(drop_age_costs_query)
    
    create_age_costs_query = text("""
    CREATE TABLE age_based_costs (
        age_costs_id SERIAL PRIMARY KEY,
        age_group VARCHAR(50) NOT NULL,
        disease_id INT NOT NULL,
        disease_name VARCHAR(100) NOT NULL,
        patient_count INT,
        visit_count INT,
        medical_care_cost_total NUMERIC,
        insurance_payment NUMERIC,
        patient_payment NUMERIC,
        patient_payment_per_visit NUMERIC,
        patient_payment_per_patient NUMERIC,
        year INT,
        rank INT,
        FOREIGN KEY (disease_id) REFERENCES diseases(disease_id)
    );
    """)
    execute_query(create_age_costs_query)
    print("age_based_costs 테이블이 생성되었습니다.")

# 병원 종별 진료비 추출
def extract_hospital_based_costs():
    engine = get_db_engine()

    query =text("""
    SELECT
        ht.hospital_type_id,
        rd.hospital_type,
        d.disease_id,
        d.disease_name,
        rd.patient_count,
        rd.visit_count,
        rd.medical_care_cost_total,
        rd.insurance_payment,
        rd.patient_payment,
        rd.patient_payment_per_visit,
        rd.patient_payment_per_patient,
        rd.year
    FROM
        raw_medical_data rd
    JOIN
        hospital_types ht ON rd.hospital_type = ht.hospital_type
    JOIN 
        diseases d ON rd.disease_code = d.disease_code
    ORDER BY
        ht.hospital_type_id, d.disease_id, rd.year
    """)

    df = pd.read_sql_query(query, engine)
    print(f"병원 종별 데이터 : {len(df)}개")

    return df

# 지역별 진료비 추출
def extract_region_based_costs():
    engine = get_db_engine()

    query = text("""
    SELECT
        r.region_id,
        rd.region_name,
        d.disease_id,
        d.disease_name,
        rd.patient_count,
        rd.visit_count,
        rd.medical_care_cost_total,
        rd.insurance_payment,
        rd.patient_payment,
        rd.patient_payment_per_visit,
        rd.patient_payment_per_patient,
        rd.year
    FROM
        raw_medical_data rd
    JOIN
        regions r ON rd.region_name = r.region_name
    JOIN
        diseases d ON rd.disease_code = d.disease_code
    ORDER BY
        r.region_id, d.disease_id, rd.year
    """)

    df = pd.read_sql_query(query, engine)
    print(f"지역별 데이터: {len(df)} 행 추출")
    
    return df

# 연령 구간 별 진료비 데이터
def extract_age_based_costs():
    engine = get_db_engine()
    
    query = text("""
    SELECT 
        rd.age_group,
        d.disease_id,
        d.disease_name,
        rd.patient_count,
        rd.visit_count,
        rd.medical_care_cost_total,
        rd.insurance_payment,
        rd.patient_payment,
        rd.patient_payment_per_visit,
        rd.patient_payment_per_patient,
        rd.year,
        rd.rank
    FROM 
        raw_medical_data rd
    JOIN 
        diseases d ON rd.disease_code = d.disease_code
    WHERE
        rd.age_group IS NOT NULL
    ORDER BY 
        rd.age_group, d.disease_id, rd.year
    """)
    
    df = pd.read_sql_query(query, engine)
    print(f"연령별 데이터: {len(df)} 행 추출")
    
    return df

# 데이터 요약 정보 출력
def show_summary(table_name, df, group_columns):
    print(f"\n{table_name} 데이터 요약")

    try:
        summary = df.groupby(group_columns + ['year'])[['patient_count', 'medical_care_cost_total']].sum()
        print(summary.head())
    except Exception as e:
        print(f"요약 정보 오류: {e}")


def process_all_data():
    try:
        # 1. 의존성 있는 테이블 먼저 삭제
        print("=======의존성 있는 테이블 먼저 삭제======")
        # 분석 테이블 삭제
        execute_query(text("DROP TABLE IF EXISTS hospital_based_costs;"))
        execute_query(text("DROP TABLE IF EXISTS region_based_costs;"))
        execute_query(text("DROP TABLE IF EXISTS age_based_costs;"))
        
        # 기본 테이블 삭제
        execute_query(text("DROP TABLE IF EXISTS hospital_types;"))
        execute_query(text("DROP TABLE IF EXISTS regions;"))


        # 1. 기본 테이블 생성
        print("=======테이블 생성======")
        create_hospital_types_table()
        create_regions_table()
        create_costs_table()

        print("======데이터 추출 및 적재======")
        # 2. 병원 종별 진료비 데이터
        hospital_costs_df = extract_hospital_based_costs()
        show_summary('hospital_based_costs', hospital_costs_df, ['hospital_type'])
        insert_dataframe(hospital_costs_df, 'hospital_based_costs', if_exists='append')

        # 3. 지역별 진료비 데이터
        region_costs_df = extract_region_based_costs()
        show_summary('region_based_costs', region_costs_df, ['region_name'])
        insert_dataframe(region_costs_df, 'region_based_costs', if_exists='append')

        # 4. 연령별 진료비 데이터
        age_costs_df = extract_age_based_costs()
        show_summary('age_based_costs', age_costs_df, ['age_group'])
        insert_dataframe(age_costs_df, 'age_based_costs', if_exists='append')

        print("\n모든 데이터 추출 및 저장이 완료되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    process_all_data()