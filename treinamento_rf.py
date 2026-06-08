import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from salvar_classificador import salvar_classificador

def treinar_rf(X_treino, y_treino, preprocessador):
    n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=20)]
    max_depth = [int(x) for x in np.linspace(start=5, stop=50, num=10)] + [None]
    criterion = ['gini', 'entropy']
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    max_features = ['sqrt', 'log2']
    n_jobs = -1
    random_state = 42
    verbose = 2
    cv = 5
    n_iter = 20

    rf_grid = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'criterion': criterion,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
    }

    classificador_rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

    hiperparametros_rf = RandomizedSearchCV(
        estimator=classificador_rf,
        param_distributions=rf_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state
    )

    hiperparametros_rf.fit(X_treino, y_treino)

    melhor_rf = hiperparametros_rf.best_estimator_

    salvar_classificador(melhor_rf, 'classificador_rf')
    salvar_classificador(preprocessador, 'preprocessador')

    return melhor_rf