import pandas as pd

def carregar_dados():
    return pd.read_csv('./datasets/default_of_credit_card_clients.csv', sep=';')