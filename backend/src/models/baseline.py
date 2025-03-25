# 베이스라인 모델 (선형회귀)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import X_train_processed, X_val_processed, feature_names
from train_test_split import y_train, y_val

nan_count = np.sum(pd.isna(y_train))
print(f"개수: {nan_count}")
print(f"y_train의 전체 크기: {len(y_train)}")


# 베이스라인 선형 회귀 모델 학습
def train_baseline_model(X_train_processed, y_train, X_val_processed, y_val):
    """
    선형 회귀 베이스라인 모델 학습 및 평가
    
    Parameters:
    -----------
    X_train_processed : 전처리된 훈련 데이터 특성
    y_train : 훈련 데이터 타겟 (진료비)
    X_val_processed : 전처리된 검증 데이터 특성
    y_val : 검증 데이터 타겟 (진료비)
    
    Returns:
    --------
    model : 훈련된 선형 회귀 모델
    metrics : 성능 지표 딕셔너리

    평가지표:
    --------
    RMSE(Root Mean Squared Error) : 예측 오차의 제곱 평균의 제곱근, 작을수록 좋다.
    MAE(Mean Absolute Error) : 절대 오차의 평균, 작을수록 좋다.
    R^2(결정계수): 모델이 설명하는 분산의 비율, 1에 가까울수록 좋다.
    MAPE(Mean Absolute Percentage Error) : 상대적 오차의 평균, 작을수록 좋다.
    """
    # 선형 회귀 모델 초기화
    model = LinearRegression()
    
    # 모델 학습 : 두 데이터 간의 관계를 학습
    model.fit(X_train_processed, y_train)
    
    # 훈련 세트 예측 : 훈련 데이터에 대한 모델의 예측값을 계산 -> 모델이 학습 데이터를 얼마나 잘 맞추는지 평가하는데 사용
    y_train_pred = model.predict(X_train_processed)
    
    # 검증 세트 예측 : 검증 데이터에 대한 모델의 예측값을 계산 -> 모델이 처음보는 데이터에 대해 얼마나 잘 맞추는지 평가하는데 사용
    y_val_pred = model.predict(X_val_processed)
    
    # 성능 지표 계산
    """
    RMSE(Root Mean Squared Error, 평균 제곱근 오차)
    - 실제 값과 예측값 차이의 제곱을 평균한 후 제곱근을 취한다.
    - 오차의 크기에 더 민감한 지표로, 큰 오차에 더 많은 패널티를 부여한다.
    - 단위가 원본 데이터와 동일하여 해석이 직관적이다.
    - 값이 작을수록 모델 성능이 좋다.

    MAE(Mean Absolute Error, 평균 절대 오차)
    - 실제값과 예측값 차이의 절대값을 평균한다.
    - 모든 오차에 동일한 가중치를 부여한다.
    - 원본 데이터와 단위가 같다.
    - 값이 작을수록 모델 성능이 좋다.

    R^2(R-squared, 결정계수)
    - 모델이 데이터 분산을 얼마나 잘 설명하는지 나타낸다.
    - 1에 가까울수록 모델이 데이터를 잘 설명한다.
    - 음수값이 나오면 모델이 평균값보다도 예측을 못한다는 의미이다.
    - 일반적으로 0.7 이상이면 괜찮은 모델이다.

    MAPE(Mean Absolute Percentage Error, 평균 절대 백분율 오차)
    - 실제값과 예측값의 차이를 실제값에 대한 백분율로 표현한 후 평균을 계산한다.
    - 상대적 오차를 측정하므로 다른 스케일의 데이터셋 간 비교가 가능하다.
    - MAPE가 10%라면 평균적으로 예측이 실제값과 10% 차이가난다는 의미이다.
    - 값이 작을수록 모델 성능이 좋다.
    
    - 훈련 데이터에서의 성능 : 모델이 학습한 데이터를 얼마나 잘 맞추는지 확인
    - 검증 데이터에서의 성능 : 모델이 처음보는 데이터에 얼마나 잘 일반화되었는지 확인
    => 두 데이터셋 간의 성능 차이가 크면 overfitting(과적합), 성능 차이가 낮으면 underfitting(과소적합)
    """
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100
    
    # 결과 저장
    metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_mape': train_mape,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'val_mape': val_mape
    }
    
    print("\n===== 베이스라인 모델 (선형 회귀) 성능 =====")
    print(f"훈련 RMSE: {train_rmse:.2f}")
    print(f"훈련 MAE: {train_mae:.2f}")
    print(f"훈련 R²: {train_r2:.4f}")
    print(f"훈련 MAPE: {train_mape:.2f}%")
    print("-" * 40)
    print(f"검증 RMSE: {val_rmse:.2f}")
    print(f"검증 MAE: {val_mae:.2f}")
    print(f"검증 R²: {val_r2:.4f}")
    print(f"검증 MAPE: {val_mape:.2f}%")

    """
    계수(coeffiecients) : 각 특성이 예측결과에 미치는 영향력의 크기과 방향을 나타내는 수치
        - 양의 계수 : 해당 특성이 존재할 때 진료비가 증가함
        - 음의 계수 : 해당 특성이 존재할 때 진료비가 감소함.
        - 계수의 절대값이 클수록 진료비에 미치는 영향이 큼.
    잔차(residual): 분석으로 모델 가정 검증
    """
    # 모델 계수 시각화 (상위 10개 긍정/부정 영향 요소)
    """
    hasattr : 객체가 특정 속성을 가지고 있는지 확인하는 함수
    coef_ : 선형회귀모델은 각 특성에 대한 가중치를 coef_ 속성에 저장
    ** 일부 모델은 이 속성이 없을 수 있으므로 검사가 필요 
    """
    if hasattr(model, 'coef_'):
        # 계수 df 생성 : 모델의 계수와 특성이름을 데이터프레임으로 만든다.
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_   
            # 계수 값을 기준으로 내림차순 정렬하여 영향력이 큰 특성부터 표시
        }).sort_values('Coefficient', ascending=False)
        
        plt.figure(figsize=(12, 8))
        
        # 상위 10개 특성 (긍정적 영향)
        plt.subplot(2, 1, 1)
        sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(10))
        plt.title('Top 10 Positive Impact Features')
        plt.tight_layout()
        
        # 하위 10개 특성 (부정적 영향)
        plt.subplot(2, 1, 2)
        sns.barplot(x='Coefficient', y='Feature', data=coef_df.tail(10))
        plt.title('Top 10 Negative Impact Features')
        plt.tight_layout()
        
        plt.savefig('baseline_feature_importance.png')
        plt.close()  # 그래프 객체를 닫아 메모리 해제
    
    # 예측 vs 실제 산점도 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('Actual Cost')
    plt.ylabel('Predicted Cost')
    plt.title('Baseline Model: Predicted vs Actual Cost (Validation Set)')
    plt.tight_layout()
    plt.savefig('baseline_prediction_scatter.png')
    plt.close()
    
    return model, metrics

# 모델 저장
def save_model(model, filename='baseline_model.pkl'):
    """훈련된 모델 저장"""
    import joblib
    joblib.dump(model, filename)
    print(f"모델이 {filename}으로 저장되었습니다.")

# 실행
if __name__ == "__main__":
    baseline_model, baseline_metrics = train_baseline_model(X_train_processed, y_train, X_val_processed, y_val)
    save_model(baseline_model)