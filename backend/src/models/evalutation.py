# 평가 지표 보류

import numbers as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def evaluate_models(models, X_test, y_test, feature_names=None):
    """
    여러 모델을 테스트 데이터로 평가하고 결과를 비교합니다.
    
    Parameters:
    -----------
    models : dict
        모델 이름과 모델 객체의 딕셔너리
    X_test : array-like
        테스트 데이터 특성
    y_test : array-like
        테스트 데이터 타겟 (실제 진료비)
    feature_names : list, optional
        특성 이름 목록 (특성 중요도 시각화에 사용)
        
    Returns:
    --------
    results_df : DataFrame
        각 모델의 평가 지표를 담은 데이터프레임
    """
    results = []
    
    for model_name, model in models.items():
        # 모델 예측
        y_pred = model.predict(X_test)
        
        # 성능 지표 계산
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results.append({
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        })
        
        print(f"\n===== {model_name} 모델 테스트 성능 =====")
        print(f"테스트 RMSE: {rmse:.2f}")
        print(f"테스트 MAE: {mae:.2f}")
        print(f"테스트 R²: {r2:.4f}")
        print(f"테스트 MAPE: {mape:.2f}%")
        
        # 특성 중요도 시각화 (트리 기반 모델용)
        if feature_names is not None and hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title(f'{model_name} - 테스트 데이터 기준 상위 20개 중요 특성')
            plt.tight_layout()
            plt.savefig(f'{model_name.lower().replace(" ", "_")}_test_feature_importance.png')
            plt.close()
        
        # 예측 vs 실제 산점도 시각화
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('실제 진료비')
        plt.ylabel('예측 진료비')
        plt.title(f'{model_name}: 예측 vs 실제 진료비 (테스트 세트)')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_test_prediction_scatter.png')
        plt.close()
    
    # 결과 비교를 위한 데이터프레임 생성
    results_df = pd.DataFrame(results)
    
    # 모델 성능 비교 시각화
    plt.figure(figsize=(12, 8))
    
    # RMSE 비교
    plt.subplot(2, 2, 1)
    sns.barplot(x='model_name', y='rmse', data=results_df)
    plt.title('테스트 RMSE')
    plt.xticks(rotation=45)
    
    # MAE 비교
    plt.subplot(2, 2, 2)
    sns.barplot(x='model_name', y='mae', data=results_df)
    plt.title('테스트 MAE')
    plt.xticks(rotation=45)
    
    # R² 비교
    plt.subplot(2, 2, 3)
    sns.barplot(x='model_name', y='r2', data=results_df)
    plt.title('테스트 R²')
    plt.xticks(rotation=45)
    
    # MAPE 비교
    plt.subplot(2, 2, 4)
    sns.barplot(x='model_name', y='mape', data=results_df)
    plt.title('테스트 MAPE (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_test_comparison.png')
    plt.close()
    
    print("\n===== 모델 테스트 성능 비교 =====")
    print(results_df)
    
    # 최적 모델 찾기
    best_model_idx = results_df['rmse'].idxmin()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    
    print(f"\n최적 모델: {best_model_name}")
    print(f"테스트 RMSE: {results_df.loc[best_model_idx, 'rmse']:.2f}")
    print(f"테스트 R²: {results_df.loc[best_model_idx, 'r2']:.4f}")
    
    return results_df


def evaluate_single_model(model, X_test, y_test, model_name, feature_names=None):
    """
    단일 모델을 테스트 데이터로 평가합니다.
    
    Parameters:
    -----------
    model : 모델 객체
        평가할 모델
    X_test : array-like
        테스트 데이터 특성
    y_test : array-like
        테스트 데이터 타겟 (실제 진료비)
    model_name : str
        모델 이름
    feature_names : list, optional
        특성 이름 목록 (특성 중요도 시각화에 사용)
        
    Returns:
    --------
    metrics : dict
        평가 지표를 담은 딕셔너리
    """
    # 모델 예측
    y_pred = model.predict(X_test)
    
    # 성능 지표 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    metrics = {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }
    
    print(f"\n===== {model_name} 모델 테스트 성능 =====")
    print(f"테스트 RMSE: {rmse:.2f}")
    print(f"테스트 MAE: {mae:.2f}")
    print(f"테스트 R²: {r2:.4f}")
    print(f"테스트 MAPE: {mape:.2f}%")
    
    # 특성 중요도 시각화 (트리 기반 모델용)
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'{model_name} - 테스트 데이터 기준 상위 20개 중요 특성')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_test_feature_importance.png')
        plt.close()
    
    # 예측 vs 실제 산점도 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('실제 진료비')
    plt.ylabel('예측 진료비')
    plt.title(f'{model_name}: 예측 vs 실제 진료비 (테스트 세트)')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_test_prediction_scatter.png')
    plt.close()
    
    return metrics

def analyze_residuals(model, X_test, y_test, model_name):
    """
    모델의 잔차를 분석합니다.
    
    Parameters:
    -----------
    model : 모델 객체
        평가할 모델
    X_test : array-like
        테스트 데이터 특성
    y_test : array-like
        테스트 데이터 타겟 (실제 진료비)
    model_name : str
        모델 이름
    """
    # 모델 예측
    y_pred = model.predict(X_test)
    
    # 잔차 계산
    residuals = y_test - y_pred
    
    # 잔차 분포 시각화
    plt.figure(figsize=(12, 10))
    
    # 잔차 히스토그램
    plt.subplot(2, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title('잔차 분포')
    plt.xlabel('잔차')
    
    # 잔차 vs 예측값
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('잔차 vs 예측값')
    plt.xlabel('예측값')
    plt.ylabel('잔차')
    
    # 실제값 vs 예측값
    plt.subplot(2, 2, 3)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('실제값 vs 예측값')
    plt.xlabel('실제값')
    plt.ylabel('예측값')
    
    # QQ 플롯 (정규성 확인)
    plt.subplot(2, 2, 4)
    import scipy.stats as stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('잔차 QQ 플롯')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_residual_analysis.png')
    plt.close()
    
    # 잔차 통계
    residual_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals)
    }
    
    print("\n===== 잔차 통계 =====")
    for stat, value in residual_stats.items():
        print(f"{stat}: {value:.4f}")
    
    return residual_stats

def load_and_evaluate_models(model_files, X_test, y_test, feature_names=None):
    """
    저장된 모델 파일들을 로드하여 평가합니다.
    
    Parameters:
    -----------
    model_files : dict 
        모델 이름과 파일 경로의 딕셔너리
    X_test : array-like
        테스트 데이터 특성
    y_test : array-like
        테스트 데이터 타겟 (실제 진료비)
    feature_names : list, optional
        특성 이름 목록 (특성 중요도 시각화에 사용)
        
    Returns:
    --------
    results_df : DataFrame
        각 모델의 평가 지표를 담은 데이터프레임
    """
    models = {}
    
    # 모델 로드
    for model_name, file_path in model_files.items():
        models[model_name] = joblib.load(file_path)
    
    # 모델 평가
    return evaluate_models(models, X_test, y_test, feature_names)

# 파일이 직접 실행될 때의 코드
if __name__ == "__main__":
    # 테스트 데이터 로드 (예시)
    from preprocessing import X_test_processed, feature_names
    from train_test_split import y_test
    
    # 저장된 모델 파일들
    model_files = {
        'Linear Regression': 'baseline_model.pkl',
        'Ridge Regression': 'ridge_regression_model.pkl',
        'Lasso Regression': 'lasso_regression_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    # 모델 로드 및 평가
    results = load_and_evaluate_models(model_files, X_test_processed, y_test, feature_names)