from carregar_dados import carregar_dados
from preprocessamento import preparar_dados, dividir_dados, construir_preprocessador, aplicar_smote
from treinamento_rf import treinar_rf
from treinamento_knn import treinar_knn
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from collections import Counter

def avaliar(nome, modelo, X_teste, y_teste):
    y_pred = modelo.predict(X_teste)
    acc  = accuracy_score(y_teste, y_pred)
    sens = recall_score(y_teste, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_teste, y_pred).ravel()
    esp = tn / (tn + fp)
    
    print(f'\n--- Avaliação: modelo {nome} ---')
    print(f'Acurácia:      {acc:.4f}')
    print(f'Sensibilidade: {sens:.4f}')
    print(f'Especificidade:{esp:.4f}')
    
    return acc

if __name__ == "__main__":
    # carregamento da csv
    dados = carregar_dados()

    # preparacao dos dados (X=dados reais, drop e y=target)
    X, y = preparar_dados(dados)

    # divisão de treino e teste com stratify
    X_treino, X_teste, y_treino, y_teste = dividir_dados(X, y)

    # preprocessador: fit só no treino, transform em ambos
    preprocessador = construir_preprocessador()
    preprocessador.fit(X_treino)
    X_treino = preprocessador.transform(X_treino)
    X_teste = preprocessador.transform(X_teste)

    # aplicando SMOTE nos dados de treino
    # aplicando SMOTE nos dados de treino
    print(f'Frequência das classes ANTES do SMOTE: {Counter(y_treino)}')
    X_treino, y_treino = aplicar_smote(X_treino, y_treino)
    print(f'Frequência das classes DEPOIS do SMOTE: {Counter(y_treino)}')

    print(f'\nDados preparados com SMOTE:')
    print(f'X_train shape: {X_treino.shape}')
    print(f'X_test shape: {X_teste.shape}')
    print(f'y_train shape: {y_treino.shape}')
    print(f'y_test shape: {y_teste.shape}')

    # treinamento dos modelos
    print('\nTreinando com o Random Forest:')
    rf  = treinar_rf(X_treino, y_treino, preprocessador)

    print('\nTreinando com KNN - K-Nearest Neighbors:')
    knn = treinar_knn(X_treino, y_treino)

    # avaliação
    acc_rf  = avaliar('Random Forest', rf, X_teste, y_teste)
    acc_knn = avaliar('K-Nearest Neighbors', knn, X_teste, y_teste)

    # seleção do melhor modelo
    resultados = {'Random Forest': acc_rf, 'K-Nearest Neighbors': acc_knn}
    melhor = max(resultados, key=resultados.get)
    print(f'\n>>> {melhor} é o melhor modelo (Acurácia: {resultados[melhor]:.4f})')