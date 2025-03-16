import pandas as pd
import requests
from bs4 import BeautifulSoup
from db_config import insert_dataframe
from playwright.sync_api import syn_palywright

def load_disease_codes_from_csv(file_path="disease_codes.csv"):
    disease_codes_df = pd.read_csv(file_path, encoding='utf-8-sig')

    print(f"총 {len(disease_codes_df)}개의 질병코드 가져옴")
    return disease_codes_df

def crawl_disesea_info(disease_codes_df):
    all_data = []

    # 기본 url과 헤더 설정
    base_url = "https://opendata.hira.or.kr/op/opc/olap3thDsInfoTab4.do"
    data_source = "hospital_types"

    # iterrows()는 pandas DataFrame의 메서드로, DataFrame의 각 행을 반복(iterate)할 수 있게 해주는 함수
    for index, row in disease_codes_df.iterrows():
        disease_code = row['disease_code']
        disease_name = row['disease_name']
        # print(f"크롤링 시작: {disease_code}")

        search_data = {
            'searchWrd' : disease_name,  # 검색창에 표시되는 값
            'olapCd' : disease_code,   
            'olapCdNm' :  disease_name,
            'tabGubun': 'Tab4',
            'sRvYr' : 2021,   # 심사년도 시작년  
            'eRvYr' : 2023    # 심사년도 종료년
        }

        response = requests.post(base_url, data=search_data)
        print(f"HTTP 응답상태 코드: {response.status_code}")

        soup = BeautifulSoup(response.content, 'html.parser')  # 응답받은 html 파싱
        
        print(soup.prettify())
        # 테이블 찾기 - select는 리스트를 반환하므로 select_one을 사용하거나 첫 번째 요소를 선택해야 함
        table = soup.find("div", class_="tblType02").find("table")
        # 또는 아래처럼 첫 번째 요소를 선택
        # tables = soup.select('div.tblType02 > table')
        # if tables:
        #     table = tables[0]

        # 테이블 body 추출
        rows = table.select('tbody tr')
        for tr in rows:
            category = tr.select_one('th')
            if category and '계' not in category.text:
                hospital_type = category.text.strip()

                # 각 셀 데이터 추출
                cells_data = [cell.text.strip() for cell in tr.select('td')]
                
                # 데이터 구조화
                # 2021년
                all_data.append({
                    'data_source': data_source,
                    'disease_code': disease_code,
                    'disease_name': disease_name,
                    'hospital_type': hospital_type,
                    'year': 2021,
                    'patient_count': cells_data[0].replace(',', ''),
                    'visit_count': cells_data[1].replace(',', ''),
                    'medical_care_cost_total': cells_data[3].replace(',', ''),
                    'insurance_payment': cells_data[4].replace(',', ''),
                    'patient_payment' : cells_data[3] - cells_data[4],
                    'patient_payment_per_visit': (cells_data[3] - cells_data[4]) / cells_data[1],
                    'patient_payment_per_patient': (cells_data[3] - cells_data[4]) / cells_data[0],
                })

                # 2022년
                all_data.append({
                    'data_source': data_source,
                    'disease_code': disease_code,
                    'disease_name': disease_name,
                    'hospital_type': hospital_type,
                    'year': 2022,
                    'patient_count': cells_data[5].replace(',', ''),
                    'visit_count': cells_data[6].replace(',', ''),
                    'medical_care_cost_total': cells_data[8].replace(',', ''),
                    'insurance_payment': cells_data[9].replace(',', ''),
                    'patient_payment' : cells_data[8] - cells_data[9],
                    'patient_payment_per_visit': (cells_data[8] - cells_data[9]) / cells_data[6],
                    'patient_payment_per_patient': (cells_data[8] - cells_data[9]) / cells_data[5],
                })

                # 2023년
                all_data.append({
                    'data_source': data_source,
                    'disease_code': disease_code,
                    'disease_name': disease_name,
                    'hospital_type': hospital_type,
                    'year': 2023,
                    'patient_count': cells_data[10].replace(',', ''),
                    'visit_count': cells_data[11].replace(',', ''),
                    'medical_care_cost_total': cells_data[13].replace(',', ''),
                    'insurance_payment': cells_data[14].replace(',', ''),
                    'patient_payment' : cells_data[13] - cells_data[14],
                    'patient_payment_per_visit': (cells_data[13] - cells_data[14]) / cells_data[11],
                    'patient_payment_per_patient': (cells_data[13] - cells_data[14]) / cells_data[10],
                })

    if all_data:
        result_df = pd.DataFrame(all_data)

        try:
            insert_dataframe(result_df, table_name='raw_data')
            print(f"데이터가 테이블에 저장되었습니다.")
        except Exception as e:
            print(f"데이터베이스 저장 중 오류 발생: {e}")


if __name__ == '__main__':
    disease_codes_df = load_disease_codes_from_csv()
    crawl_disesea_info(disease_codes_df=disease_codes_df)