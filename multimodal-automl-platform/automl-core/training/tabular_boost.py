import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

class TabularBoostTrainer:
    def __init__(self, algorithm: str = "xgboost", learning_rate: float = 0.05, n_estimators: int = 100, max_depth: int = 5):
        self.learning_rate = learning_rate
        if algorithm.lower() == "xgboost" and HAS_XGB:
            self.model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, eval_metric="logloss")
        else:
            self.model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

    def fit_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_val)
        probs = self.model.predict_proba(X_val)
        return self.model, {
            "accuracy": float(accuracy_score(y_val, preds)),
            "loss": float(log_loss(y_val, probs)),
            "learning_rate": self.learning_rate
        }
