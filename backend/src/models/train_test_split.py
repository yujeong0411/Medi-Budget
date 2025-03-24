import pandas as pd
from sklearn.model_selection import train_test_split
from configs.db_config import get_db_engine

engine = get_db_engine()

query = "SELECT * FROM raw_medical_data"
data = pd.read_sql_query(query, engine)

# 특성(x)과 타켓(y) 분리
X = data.drop(['patient_payment_per_patient'], axis=1)   # 타겟변수 외 모든 열을 특성으로 사용
y = data['patient_payment_per_patient']

# 데이터 분할 (70/15/15)
# 먼저 훈련(70)과 나머지(30)로 분할
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# 나머지를 검증(15)과 테스트(15)로 분할
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 분할 결과 확인
print(f"전체 데이터: {len(data)}개")
print(f"훈련 데이터: {len(X_train)}개")
print(f"검증 데이터: {len(X_val)}개")
print(f"테스트 데이터: {len(X_test)}개")



