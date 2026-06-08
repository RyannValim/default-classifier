from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def preparar_dados(dados):
    # quebra os dados em X: base com drop em ID e DEFAULT (target) e y: que é o target, renomeia a coluna target para ficar mais fácil
    dados = dados.rename(columns={'default payment next month': 'DEFAULT'})
    X = dados.drop(columns=['ID', 'DEFAULT'])
    y = dados['DEFAULT']
    
    return X, y

def dividir_dados(X, y):
    # divide os dados de treino com stratify, que separa os dados proporcionalmente com o 78/22
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def construir_preprocessador():
    # separação das colunas categóricas e numéricas
    dados_cat = ['SEX', 'EDUCATION', 'MARRIAGE']
    dados_num = ['LIMIT_BAL', 'AGE',
                 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    # instancias de scaler e encoder
    ohe_encoder = OneHotEncoder(handle_unknown='ignore')
    standard_scaler = StandardScaler()

    # necessário para as colunas categóricas
    preprocessador = ColumnTransformer([
        ('cat', ohe_encoder, dados_cat),
        ('num', standard_scaler, dados_num),
    ], remainder='passthrough')

    return preprocessador

def aplicar_smote(X_treino, y_treino):
    # aplica SMOTE nos dados de treino e antes do treinamento somente para evitar vazamento de dados (data leakage)
    smote = SMOTE(random_state=42)
    
    return smote.fit_resample(X_treino, y_treino)