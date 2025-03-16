import os
import pandas as pd
import glob
from db_config import get_db_engine, insert_dataframe

data_dir = "../../data/raw"
table_name = 'raw_data'  # 원본 테이블

# 결과를 저장할 데이터프레임 초기화
all_data = pd.DataFrame()

# 디렉토리에서 모든 엑셀파일 찾기
excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))

for file_path in excel_files:
    # basename 사용하여 파일명만 추출
    file_name = os.path.basename(file_path) 
    print("파일이름", file_name)

    # 파일 이름에서 확장자 제거 이름 추출
    base_filename = os.path.splitext(file_name)[0]

    # disease_ 부분 제거 후 나머지 나이 추출
    age_part = base_filename.replace('disease_', '')

    if '_' in age_part:
        start, end = age_part.split('_')
        age_group = f"{start}~{end}세"
    else:
        age_group = f"{age_part}세 이상"

    print(f"나이 그룹 추출: {age_group}")

    try:
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path, header=3)  # 데이터가 시작되는 행은 3행부터
        print(f"파일 {file_name} 로드 성공")

         # 전체 컬럼명 출력
        print("전체 컬럼:")
        for col in df.columns:
            print(col)


        years = [2021, 2022, 2023]
        # 년도별로 달라지는 컬럼
        year_columns = {
            2023: {
                '환자수': '환자수', 
                '내원일수': '내원일수', 
                '요양급여비용총액': '요양급여비용총액', 
                '보험자부담금': '보험자부담금'
            },
            2022: {
                '환자수.1': '환자수', 
                '내원일수.1': '내원일수', 
                '요양급여비용총액.1': '요양급여비용총액', 
                '보험자부담금.1': '보험자부담금'
            },
            2021: {
                '환자수.2': '환자수', 
                '내원일수.2': '내원일수', 
                '요양급여비용총액.2': '요양급여비용총액', 
                '보험자부담금.2': '보험자부담금'
            }
        }


        # 데이터 처리 및 통합
        processed_data = []
        for year in years:
            required_cols = list(year_columns[year].keys())
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                print(f"{year}년 데이터 처리 불가: 다음 컬럼이 없습니다 - {', '.join(missing_cols)}")
                continue

            # 필요한 컬럼만 선택하여 복사
            selected_cols = ['코드', '3단질병명', '순위'] + list(year_columns[year].keys())
            year_df = df[selected_cols].copy()

             # 해당 년도의 컬럼명 매핑
            year_df.rename(columns=year_columns[year], inplace=True)
            print(f"{year}년 데이터 컬럼명 변경 완료")
         
            # 년도 추가
            year_df['심사년도'] = year

            # 연령대 추가
            year_df['연령대'] = age_group

            # 데이터 유형 추가
            year_df['data_source'] = file_name

            # 진료비 필드 추가
            year_df['본인부담금총액'] = round(year_df['요양급여비용총액'] - year_df['보험자부담금'], 2)
            year_df['방문당_본인부담금'] = round(year_df['본인부담금총액'] / year_df['내원일수'], 2)
            year_df['환자당_본인부담금'] = round(year_df['본인부담금총액'] / year_df['환자수'], 2)

            processed_data.append(year_df)

        # 결과 데이터 프레임에 추가
        all_data = pd.concat([all_data] + processed_data, ignore_index=True)

        print(f"successfully processed {file_name}, extracted {len(year_df)}")
    except Exception as e:
        print(f"에러: {file_name}: {e}")

# 결과확인
print(f"total extracted data: {len(all_data)} rows")
print(all_data.head())

# DB 컬럼명와 일치시키기
final_column_mapping = {
    '심사년도': 'year',
    '3단질병명': 'disease_name',
    '코드': 'disease_code',
    '순위': 'rank',
    '요양급여비용총액': 'medical_care_cost_total',
    '보험자부담금': 'insurance_payment',
    '본인부담금총액': 'patient_payment',
    '내원일수': 'visit_count',
    '환자수': 'patient_count',
    '연령대': 'age_group',
    '방문당_본인부담금': 'patient_payment_per_visit',
    '환자당_본인부담금' : 'patient_payment_per_patient',
    '데이터소스': 'data_source'
}

all_data.rename(columns=final_column_mapping, inplace=True)

# DB에 저장
try:
    insert_dataframe(all_data, table_name, if_exists='append')
    print(f"데이터가 {table_name} 테이블에 저장되었습니다.")
except Exception as e:
    print(f"데이터베이스 저장 중 오류 발생: {e}")
        

# CSV로도 저장
output_path = os.path.join(data_dir, "extracted_disease_data.csv")
all_data.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"data saved to {output_path}")