# 특성 전처리
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from train_test_split import X_train, X_val, X_test

# print("X_train 컬럼 이름:", X_train.columns.tolist())

# 전처리를 위한 변수 타입 분류
numerical_features = [
    'patient_count', 'visit_count', 'medical_care_cost_total', 'insurance_payment', 'rank', 'patient_payment_per_visit',
    'year'
]

categorical_features = [
    'hospital_type', 'region_name', 'disease_name', 'disease_code', 'age_group'
]

# 범주형 변수에 대한 전처리 (원-핫 인코딩)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 결측치는 최빈값으로 대체
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse_output=False))  # 원-핫 인코딩 변환
])

# 수치형 변수에 대한 전처리 (스케일링)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),   # 결측치는 중앙값으로 대체
    ('scaler', StandardScaler())   # 표준화 스케일링으로 변수 정규화
])

# ColumnTransformer를 사용하여 모든 변수 전처리 통합
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 전처리 적용 
# 훈련 데이터에 맞추고 반환
# 모델 훈련: 전처리된 데이터(X_train_processed, y_train)로 모델 학습
X_train_processed =  preprocessor.fit_transform(X_train)   # fit_transform으로 전처리기를 훈련시키고 적용
# 검증 및 테스트 데이터는 fit_transform이 아닌 transform만 적용
# transform으로 이미 학습된 파라미터를 사용해 적용
# 모델 평가: 전처리된 검증/테스트 데이터로 모델 성능 평가
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# 전처리된 특성의 이름 가져오기
# 원-핫 인코딩 후 생성된 특성 이름을 가져오는 함수
# 모델 해석 시 어떤 특성이 중요한지 이해하는데 필요
def get_feature_names(column_transformer):
    output_features = []

    for name, pipe, features in column_transformer.transformers_:
        if name == 'drop' or pipe =='drop':
            continue

        if isinstance(pipe, Pipeline):
            transformer = pipe.steps[-1][1]
        else:
            transformer = pipe

        if hasattr(transformer, 'get_feature_names_out'):
            if isinstance(features, (list, np.ndarray)):
                output_features.extend(
                    transformer.get_feature_names_out(features)
                )
            else:
                output_features.extend(
                    transformer.get_feature_names_out()
                )
        else:
            output_features.extend(features)

    return output_features

feature_names = get_feature_names(preprocessor)
print(f"전처리 후 특성 수: {len(feature_names)}개")
print(f"처음 10개 특성: {feature_names[:10]}")

# 전처리 파이프라인 저장
import joblib
joblib.dump(preprocessor, 'preprocessor.pk1')
