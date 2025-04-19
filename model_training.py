from xgboost import XGBRegressor
from pyswarm import pso
from sklearn.metrics import mean_absolute_error

def tune_xgb(X_train, y_train, X_val, y_val, seed=42):
    def fitness(params):
        lr, md, ne, ss = params
        model = XGBRegressor(
            learning_rate=lr,
            max_depth=int(round(md)),
            n_estimators=int(round(ne)),
            subsample=ss,
            objective='reg:squarederror',
            random_state=seed,
            verbosity=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)
    lb = [0.01,3,50,0.5]
    ub = [0.3,10,300,1.0]
    best, _ = pso(fitness, lb, ub, swarmsize=10, maxiter=15)
    return XGBRegressor(
        learning_rate=best[0],
        max_depth=int(round(best[1])),
        n_estimators=int(round(best[2])),
        subsample=best[3],
        objective='reg:squarederror',
        random_state=seed
    )
