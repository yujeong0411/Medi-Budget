# 베이스라인 모델 (선형회귀)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import X_train_processed, X_val_processed
from train_test_split import X_train, X_val, y_train, y_val

# 베이스라인 선형 회귀 모델 학습
def train_baseline_model(X_train, y_train, X_val, y_val):
    """
    선형 회귀 베이스라인 모델 학습 및 평가
    
    Parameters:
    -----------
    X_train : 훈련 데이터 특성
    y_train : 훈련 데이터 타겟 (진료비)
    X_val : 검증 데이터 특성
    y_val : 검증 데이터 타겟 (진료비)
    
    Returns:
    --------
    model : 훈련된 선형 회귀 모델
    metrics : 성능 지표 딕셔너리
    """
    # 선형 회귀 모델 초기화
    model = LinearRegression()
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 훈련 세트 예측
    y_train_pred = model.predict(X_train)
    
    # 검증 세트 예측
    y_val_pred = model.predict(X_val)
    
    # 성능 지표 계산
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
    
    # 모델 계수 시각화 (상위 10개 긍정/부정 영향 요소)
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False)
        
        plt.figure(figsize=(12, 8))
        
        # 상위 10개 특성 (긍정적 영향)
        plt.subplot(2, 1, 1)
        sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(10))
        plt.title('상위 10개 긍정적 영향 특성')
        plt.tight_layout()
        
        # 하위 10개 특성 (부정적 영향)
        plt.subplot(2, 1, 2)
        sns.barplot(x='Coefficient', y='Feature', data=coef_df.tail(10))
        plt.title('상위 10개 부정적 영향 특성')
        plt.tight_layout()
        
        plt.savefig('baseline_feature_importance.png')
        plt.close()
    
    # 예측 vs 실제 산점도 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('실제 진료비')
    plt.ylabel('예측 진료비')
    plt.title('베이스라인 모델: 예측 vs 실제 진료비 (검증 세트)')
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
    # 여기에 데이터 로드 및 전처리 코드 추가
    # 이전 코드에서 전처리된 데이터를 사용한다고 가정
    
    baseline_model, baseline_metrics = train_baseline_model(X_train_processed, y_train, X_val_processed, y_val)
    save_model(baseline_model)