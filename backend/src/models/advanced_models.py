# 고급 모델
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

# 모델 평가 함수
def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    모델 학습 및 평가
    
    Parameters:
    -----------
    model : 학습할 모델 객체
    X_train, y_train : 훈련 데이터
    X_val, y_val : 검증 데이터
    model_name : 모델 이름 (문자열)
    
    Returns:
    --------
    model : 훈련된 모델
    metrics : 성능 지표 딕셔너리
    """
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train)
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
        'model_name': model_name,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_mape': train_mape,
        'val_rmse': val_rmse,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'val_mape': val_mape
    }
    
    print(f"\n===== {model_name} 모델 성능 =====")
    print(f"훈련 RMSE: {train_rmse:.2f}")
    print(f"훈련 MAE: {train_mae:.2f}")
    print(f"훈련 R²: {train_r2:.4f}")
    print(f"훈련 MAPE: {train_mape:.2f}%")
    print("-" * 40)
    print(f"검증 RMSE: {val_rmse:.2f}")
    print(f"검증 MAE: {val_mae:.2f}")
    print(f"검증 R²: {val_r2:.4f}")
    print(f"검증 MAPE: {val_mape:.2f}%")
    
    # 특성 중요도 시각화 (트리 기반 모델용)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'{model_name} - 상위 20개 중요 특성')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.close()
    
    # 예측 vs 실제 산점도 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('실제 진료비')
    plt.ylabel('예측 진료비')
    plt.title(f'{model_name}: 예측 vs 실제 진료비 (검증 세트)')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_prediction_scatter.png')
    plt.close()
    
    return model, metrics

# 모든 모델 훈련 및 평가
def train_all_models(X_train, y_train, X_val, y_val):
    """여러 모델 훈련 및 평가"""
    
    # 모델 정의
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }
    
    # 모델 결과 저장
    results = []
    trained_models = {}
    
    # 각 모델 훈련
    for model_name, model in models.items():
        print(f"\n{'-'*50}\n훈련 중: {model_name}\n{'-'*50}")
        trained_model, metrics = evaluate_model(
            model, X_train, y_train, X_val, y_val, model_name
        )
        
        # 결과 저장
        results.append(metrics)
        trained_models[model_name] = trained_model
        
        # 모델 저장
        joblib.dump(trained_model, f"{model_name.lower().replace(' ', '_')}_model.pkl")
    
    # 결과 비교
    results_df = pd.DataFrame(results)
    
    # 모델 성능 비교 시각화
    plt.figure(figsize=(12, 8))
    
    # RMSE 비교
    plt.subplot(2, 2, 1)
    sns.barplot(x='model_name', y='val_rmse', data=results_df)
    plt.title('검증 RMSE')
    plt.xticks(rotation=45)
    
    # MAE 비교
    plt.subplot(2, 2, 2)
    sns.barplot(x='model_name', y='val_mae', data=results_df)
    plt.title('검증 MAE')
    plt.xticks(rotation=45)
    
    # R² 비교
    plt.subplot(2, 2, 3)
    sns.barplot(x='model_name', y='val_r2', data=results_df)
    plt.title('검증 R²')
    plt.xticks(rotation=45)
    
    # MAPE 비교
    plt.subplot(2, 2, 4)
    sns.barplot(x='model_name', y='val_mape', data=results_df)
    plt.title('검증 MAPE (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    print("\n===== 모델 성능 비교 =====")
    comparison_metrics = ['val_rmse', 'val_mae', 'val_r2', 'val_mape']
    print(results_df[['model_name'] + comparison_metrics])
    
    # 최적 모델 찾기
    best_model_idx = results_df['val_rmse'].idxmin()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    
    print(f"\n최적 모델: {best_model_name}")
    print(f"검증 RMSE: {results_df.loc[best_model_idx, 'val_rmse']:.2f}")
    print(f"검증 R²: {results_df.loc[best_model_idx, 'val_r2']:.4f}")
    
    return trained_models, results_df

# 실행
if __name__ == "__main__":
    # 여기에 데이터 로드 및 전처리 코드 추가
    # 이전 코드에서 전처리된 데이터를 사용한다고 가정
    
    models, results = train_all_models(X_train_processed, y_train, X_val_processed, y_val)