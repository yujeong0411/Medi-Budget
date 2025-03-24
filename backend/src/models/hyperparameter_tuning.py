# 하이퍼 파라미터 튜닝
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from scipy.stats import randint, uniform

# 하이퍼파라미터 튜닝 함수
def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    RandomizedSearchCV를 사용하여 모델의 하이퍼파라미터 튜닝
    
    Parameters:
    -----------
    X_train, y_train : 훈련 데이터
    X_val, y_val : 검증 데이터
    
    Returns:
    --------
    tuned_models : 튜닝된 모델 딕셔너리
    tuning_results : 튜닝 결과 DataFrame
    """
    # 교차 검증 설정
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 모델별 하이퍼파라미터 탐색 공간
    param_spaces = {
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': uniform(0.01, 10.0),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'fit_intercept': [True, False]
            }
        },
        'Lasso': {
            'model': Lasso(),
            'params': {
                'alpha': uniform(0.001, 1.0),
                'fit_intercept': [True, False],
                'max_iter': randint(1000, 3000),
                'selection': ['cyclic', 'random']
            }
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(5, 30),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['auto', 'sqrt', 'log2']
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42),
            'params': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'gamma': uniform(0, 1),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1)
            }
        }
    }
    
    # 결과 저장
    tuned_models = {}
    tuning_results = []
    
    # 각 모델 튜닝
    for model_name, config in param_spaces.items():
        print(f"\n{'-'*50}\n하이퍼파라미터 튜닝 중: {model_name}\n{'-'*50}")
        
        # RandomizedSearchCV 설정
        random_search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=30,  # 탐색 횟수
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,  # 모든 코어 사용
            verbose=1,
            random_state=42
        )
        
        # 모델 학습
        random_search.fit(X_train, y_train)
        
        # 최적 모델
        best_model = random_search.best_estimator_
        
        # 검증 데이터로 성능 평가
        y_val_pred = best_model.predict(X_val)
        
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        val_mape = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100
        
        # 모델 저장
        tuned_models[model_name] = best_model
        joblib.dump(best_model, f"tuned_{model_name.lower().replace(' ', '_')}_model.pkl")
        
        # 결과 저장
        tuning_results.append({
            'model_name': model_name,
            'best_params': random_search.best_params_,
            'best_score': -random_search.best_score_,  # 음수 점수를 양수로 변환
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'val_mape': val_mape
        })
        
        print(f"\n{model_name} 최적 하이퍼파라미터:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\n{model_name} 검증 성능:")
        print(f"  RMSE: {val_rmse:.2f}")
        print(f"  MAE: {val_mae:.2f}")
        print(f"  R²: {val_r2:.4f}")
        print(f"  MAPE: {val_mape:.2f}%")
        
        # 학습 곡선 시각화 (가능한 경우)
        if hasattr(random_search, 'cv_results_'):
            results = pd.DataFrame(random_search.cv_results_)
            results['mean_test_score'] = -results['mean_test_score']  # 음수 점수를 양수로 변환
            
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(results)), results['mean_test_score'], alpha=0.8)
            plt.plot(results['mean_test_score'].rolling(window=3).mean(), 'r-')
            plt.xlabel('Iteration')
            plt.ylabel('RMSE (CV)')
            plt.title(f'{model_name} 하이퍼파라미터 탐색 과정')
            plt.grid(True)
            plt.savefig(f'{model_name.lower().replace(" ", "_")}_tuning_curve.png')
            plt.close()
    
    # 모델 성능 비교
    tuning_results_df = pd.DataFrame(tuning_results)
    
    # 모델 성능 비교 시각화
    plt.figure(figsize=(12, 8))
    
    # RMSE 비교
    plt.subplot(2, 2, 1)
    sns.barplot(x='model_name', y='val_rmse', data=tuning_results_df)
    plt.title('튜닝된 모델 검증 RMSE')
    plt.xticks(rotation=45)
    
    # MAE 비교
    plt.subplot(2, 2, 2)
    sns.barplot(x='model_name', y='val_mae', data=tuning_results_df)
    plt.title('튜닝된 모델 검증 MAE')
    plt.xticks(rotation=45)
    
    # R² 비교
    plt.subplot(2, 2, 3)
    sns.barplot(x='model_name', y='val_r2', data=tuning_results_df)
    plt.title('튜닝된 모델 검증 R²')
    plt.xticks(rotation=45)
    
    # MAPE 비교
    plt.subplot(2, 2, 4)
    sns.barplot(x='model_name', y='val_mape', data=tuning_results_df)
    plt.title('튜닝된 모델 검증 MAPE (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savef