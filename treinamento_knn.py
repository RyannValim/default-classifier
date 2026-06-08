from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from salvar_classificador import salvar_classificador

def treinar_knn(X_treino, y_treino):
    n_neighbors = [3, 5, 7, 9, 11, 15, 21]
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    random_state = 42
    n_iter = 10
    cv = 5
    verbose = 2
    n_jobs = -1

    knn_grid = {
        'n_neighbors': n_neighbors,
        'weights': weights,
        'metric': metric,
    }

    classificador_knn = KNeighborsClassifier()

    hiperparametros_knn = RandomizedSearchCV(
        estimator=classificador_knn,
        param_distributions=knn_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state
    )

    hiperparametros_knn.fit(X_treino, y_treino)

    melhor_knn = hiperparametros_knn.best_estimator_

    salvar_classificador(melhor_knn, 'classificador_knn')

    return melhor_knn