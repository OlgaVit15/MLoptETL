import pandas as pd
from tqdm import tqdm
import warnings

from model_ml.test.CatBoost.strategy_classifier import StrategyCascadeClassifier

# Игнорировать конкретное предупреждение о именах признаков
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from model_ml.test.CatBoost.mem_strategy_cascade import ImpalaOptimizationCascade
from model_ml.test.LGBM.lgbm_mem_strategy_cascade import LGBMImpalaOptimizationCascade
from model_ml.test.XGboost.xgb_mem_strategy_cascade import XGBImpalaOptimizationCascade
from model_ml.test.ml_cycle import load_and_clean_data

xgb = XGBImpalaOptimizationCascade()
lgb = LGBMImpalaOptimizationCascade()
cb = ImpalaOptimizationCascade()
model_xgb = xgb.load("models/xgbimpala_cascade_20260522_085151.joblib")
model_lgb = lgb.load("models/lgbimpala_cascade_20260522_085151.joblib")
model_cb = cb.load("models/impala_cascade_20260522_085151.joblib")
test = load_and_clean_data() #pd.read_csv("ml_training_dataset.csv", sep=";")
sample = test.sample().to_dict(orient='records')[0]

for _ in tqdm(range(3000)):
    model_cb.predict_s(sample)
for _ in tqdm(range(5000)):
    model_cb.predict_s(sample)

# for _ in tqdm(range(3000)):
#     model_xgb.predict_s(sample)
# for _ in tqdm(range(5000)):
#     model_xgb.predict_s(sample)
#
# for _ in tqdm(range(3000)):
#     model_lgb.predict_s(sample)
# for _ in tqdm(range(5000)):
#     model_lgb.predict_s(sample)
