import pandas as pd
from pickle import load

def carregar_classificador(nome_modelo):
    return load(open(f'./models/Default_{nome_modelo}.pkl', 'rb'))

def inferir(dados_novos, nome_modelo='classificador_rf'):
    """
    Recebe um DataFrame de clientes novos (mesmas colunas do CSV original, 
    exceto ID e DEFAULT) e retorna predição + probabilidade de default.
    """
    # carrega artefatos salvos no treinamento
    preprocessador = carregar_classificador('preprocessador')
    modelo = carregar_classificador(nome_modelo)
    
    # aplica o MESMO preprocessador do treino (só transform, sem fit)
    X = preprocessador.transform(dados_novos)
    
    # predição binária e probabilidade
    predicoes = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)
    
    # monta resultado
    resultado = pd.DataFrame({
        'predicao': predicoes,
        'prob_nao_default': probabilidades[:, 0],
        'prob_default': probabilidades[:, 1],
    })
    
    return resultado

if __name__ == "__main__":
    # exemplo: pega algumas linhas do CSV original como "novos clientes"
    from carregar_dados import carregar_dados
    
    dados = carregar_dados()
    
    # simula clientes novos pegando 5 linhas e tirando ID e DEFAULT
    novos_clientes = dados.drop(columns=['ID', 'default payment next month']).head(5)
    
    print('Clientes para inferir:')
    print(novos_clientes)
    
    resultado = inferir(novos_clientes, nome_modelo='classificador_rf')
    
    print('\nResultado da inferência:')
    print(resultado.to_string())
    
    print('\nInterpretação:')
    for i, row in resultado.iterrows():
        status = 'INADIMPLENTE' if row['predicao'] == 1 else 'ADIMPLENTE'
        score = row['prob_default'] * 100
        print(f'Cliente {i}: {status} (risco de default: {score:.2f}%)')