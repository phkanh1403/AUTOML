import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

class OptunaHyperparameterOptimizer:
    def __init__(self, n_trials: int = 10):
        self.n_trials = n_trials

    def optimize_tabular(self, X, y):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
        )
        def objective(trial: optuna.Trial):
            lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 8)
            n_est = trial.suggest_int("n_estimators", 50, 150)
            clf = GradientBoostingClassifier(learning_rate=lr, max_depth=max_depth, n_estimators=n_est)
            return float(cross_val_score(clf, X, y, cv=2, scoring="accuracy").mean())
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params, study.best_value
