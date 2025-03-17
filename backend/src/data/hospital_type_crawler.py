import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from db_config import insert_dataframe
from playwright.sync_api import sync_playwright

def load_disease_codes_from_csv(file_path="disease_codes.csv"):
    disease_codes_df = pd.read_csv(file_path, encoding='utf-8-sig')

    print(f"총 {len(disease_codes_df)}개의 질병코드 가져옴")
    return disease_codes_df

def crawl_disesea_info(disease_codes_df):
    all_data = []

    # 기본 url과 헤더 설정
    base_url = "https://opendata.hira.or.kr/op/opc/olap3thDsInfoTab4.do"
    data_source = "hospital_types"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # headless=False : 브라우저 숨김 해제
        context = browser.new_context()  # context는 독립된 쿠기, 로컬 스토리지, 세션스토리지를 가짐
        page = context.new_page()

        # iterrows()는 pandas DataFrame의 메서드로, DataFrame의 각 행을 반복(iterate)할 수 있게 해주는 함수
        for index, row in disease_codes_df.iterrows():
            disease_code = row['disease_code']
            disease_name = row['disease_name']
            print(f"크롤링 시작: {disease_code}")
        # first_row = disease_codes_df.iloc[0]
        # disease_code = first_row['disease_code']
        # disease_name = first_row['disease_name']
        
            page.goto(base_url)
            page.click('input#searchWrd')  # 입력창 클릭 
            page.fill('input#searchWrd1', disease_code)  # 코드 입력 
            page.click('a#popSearchBtn1')  # 검색 클릭 

            
            page.click('a[onclick="fnPopupSelect(1);"]')  # 해당 코드 선택

            # 심사년도 설정 및 검색
            page.select_option('select#sRvYr', '2021')
            page.select_option('select#eRvYr', '2023')
            page.click('a#searchBtn1')

            page.wait_for_load_state('networkidle')
            print("페이지 로딩 완료")

            # 페이지 콘텐츠 가져오기
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')

            # 테이블 찾기 - select는 리스트를 반환하므로 select_one을 사용하거나 첫 번째 요소를 선택해야 함
            table = soup.find("div", class_="tblType02 data webScroll").find("table")
            # 또는 아래처럼 첫 번째 요소를 선택
            # tables = soup.select('div.tblType02 > table')
            # if tables:
            #     table = tables[0]

            # 테이블 body 추출
            rows = table.select('tbody tr')
            for tr in rows:
                category = tr.select_one('th')
                # print(category)
                if category and ('계' not in category.text and disease_code not in category.text):
                    hospital_type = category.text.strip()

                    # 각 셀 데이터 추출
                    cells_data_raw = [cell.text.strip() for cell in tr.select('td')]
                    cells_data = []

                    # # 숫자로 변환 - 값이 없으면 NaN 
                    for value in cells_data_raw:
                        if value == '-':
                            cells_data.append(np.nan)
                        else:
                            cells_data.append(float(value.replace(',', '')))

                    # 데이터 구조화
                    # 2021년
                    all_data.append({
                        'data_source': data_source,
                        'disease_code': disease_code,
                        'disease_name': disease_name,
                        'hospital_type': hospital_type,
                        'year': 2021,
                        'patient_count': cells_data[0],
                        'visit_count': cells_data[1],
                        'medical_care_cost_total': cells_data[3],
                        'insurance_payment': cells_data[4],
                        'patient_payment' : cells_data[3] - cells_data[4],
                        'patient_payment_per_visit': round((cells_data[3] - cells_data[4]) / cells_data[1], 2),
                        'patient_payment_per_patient': round((cells_data[3] - cells_data[4]) / cells_data[0], 2)
                    })

                    # 2022년
                    all_data.append({
                        'data_source': data_source,
                        'disease_code': disease_code,
                        'disease_name': disease_name,
                        'hospital_type': hospital_type,
                        'year': 2022,
                        'patient_count': cells_data[5],
                        'visit_count': cells_data[6],
                        'medical_care_cost_total': cells_data[8],
                        'insurance_payment': cells_data[9],
                        'patient_payment' : cells_data[8] - cells_data[9],
                        'patient_payment_per_visit': round((cells_data[8] - cells_data[9]) / cells_data[6], 2),
                        'patient_payment_per_patient': round((cells_data[8] - cells_data[9]) / cells_data[5], 2)
                    })

                    # 2023년
                    all_data.append({
                        'data_source': data_source,
                        'disease_code': disease_code,
                        'disease_name': disease_name,
                        'hospital_type': hospital_type,
                        'year': 2023,
                        'patient_count': cells_data[10],
                        'visit_count': cells_data[11],
                        'medical_care_cost_total': cells_data[13],
                        'insurance_payment': cells_data[14],
                        'patient_payment' : cells_data[13] - cells_data[14],
                        'patient_payment_per_visit': round((cells_data[13] - cells_data[14]) / cells_data[11], 2),
                        'patient_payment_per_patient': round((cells_data[13] - cells_data[14]) / cells_data[10], 2)
                    })

    if all_data:
        result_df = pd.DataFrame(all_data)

        try:
            insert_dataframe(result_df, table_name='raw_medical_data')
            print(f"데이터가 테이블에 저장되었습니다.")
        except Exception as e:
            print(f"데이터베이스 저장 중 오류 발생: {e}")


if __name__ == '__main__':
    disease_codes_df = load_disease_codes_from_csv()
    crawl_disesea_info(disease_codes_df=disease_codes_df)