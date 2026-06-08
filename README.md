# Default Classifier — Banco AgoraVai

Sistema de classificação para prever inadimplência ( *default* ) de clientes de cartão de crédito do Banco AgoraVai, usando o dataset *Default of Credit Card Clients* (UCI / Taiwan, 2005).

O sistema treina e compara modelos de classificação, seleciona o melhor por acurácia e disponibiliza um módulo de inferência que retorna a predição binária acompanhada da distribuição de probabilidade (`predict_proba`).

---

## Estrutura do projeto

```
default-classifier/
├── datasets/
│   └── default_of_credit_card_clients.csv
├── models/                          # gerado em runtime, contém os .pkl
├── carregar_dados.py                # leitura do CSV
├── preprocessamento.py              # preparação, split, OHE+Scaler, SMOTE
├── treinamento_rf.py                # Random Forest com RandomizedSearchCV
├── treinamento_knn.py               # KNN com RandomizedSearchCV
├── salvar_classificador.py          # persistência via pickle
├── inferir.py                       # módulo de inferência (predict + predict_proba)
├── main.py                          # pipeline completo
└── README.md
```

---

## Pipeline

1. **Carregamento** (`carregar_dados.py`): leitura do CSV.
2. **Preparação** (`preparar_dados`): drop de `ID`, rename de `default payment next month` para `DEFAULT`, separação X/y.
3. **Split estratificado** (`dividir_dados`): `test_size=0.3`, `stratify=y`, `random_state=42`. A estratificação garante que treino e teste preservem a proporção real do dataset (~78/22).
4. **Preprocessador** (`construir_preprocessador`): `ColumnTransformer` com:
   * `OneHotEncoder(handle_unknown='ignore')` em `SEX`, `EDUCATION`, `MARRIAGE`
   * `StandardScaler` nas numéricas contínuas (`LIMIT_BAL`, `AGE`, `BILL_AMT1..6`, `PAY_AMT1..6`)
   * `PAY_0..PAY_6` passam direto (ordinais)
   * **Fit somente no treino** , depois `.transform()` em treino e teste — evita  *data leakage* .
5. **SMOTE** (`aplicar_smote`): aplicado  **somente no treino, depois do split e do preprocessador** . Gera amostras sintéticas da classe minoritária para balancear ~50/50. O teste permanece com a proporção real para refletir o cenário de produção.
6. **Treinamento** : Random Forest e KNN, ambos com `RandomizedSearchCV` (`cv=5`, `n_iter=10-20`) para busca de hiperparâmetros.
7. **Avaliação** : acurácia, sensibilidade e especificidade no conjunto de teste.
8. **Seleção do melhor modelo** : por acurácia, com print do vencedor.
9. **Persistência** : modelos e preprocessador salvos em `./models/Default_*.pkl` para uso em inferência.

---

## Princípios e decisões

* **SMOTE após o split** : evita que amostras sintéticas geradas a partir de pontos reais contaminem o teste ( *data leakage* ).
* **Stratify no split** : garante que treino e teste mantenham a proporção real de classes (~78% adimplente / ~22% inadimplente).
* **One-Hot em SEX/EDUCATION/MARRIAGE** : tratadas como categóricas nominais, sem ordem implícita.
* **PAY_0..PAY_6 como ordinais** : a magnitude do atraso importa, então passam direto sem encoding.
* **Preprocessador fitado só no treino** : salvo em `.pkl` e reaplicado em inferência sem retreinar.
* **Padrões do projeto** : `random_state=42`, `test_size=0.3`, `n_jobs=-1`.

---

## Como rodar

### Dependências

```bash
pip install pandas scikit-learn imbalanced-learn numpy
```

### Treinamento

```bash
python main.py
```

Saída esperada (resumo):

```
Frequência das classes ANTES do SMOTE: Counter({0: ..., 1: ...})
Frequência das classes DEPOIS do SMOTE: Counter({0: ..., 1: ...})

Treinando com o Random Forest:
...

Treinando com KNN - K-Nearest Neighbors:
...

--- Avaliação: modelo Random Forest ---
Acurácia:      0.79xx
Sensibilidade: 0.4xxx
Especificidade: 0.9xxx

--- Avaliação: modelo K-Nearest Neighbors ---
...

>>> Random Forest é o melhor modelo (Acurácia: 0.79xx)
```

### Inferência

```bash
python inferir.py
```

Saída esperada:

```
Resultado da inferência:
   predicao  prob_nao_default  prob_default
0         1          0.146679      0.853321
1         1          0.483692      0.516308
2         0          0.934345      0.065655

Cliente 0: INADIMPLENTE (risco de default: 85.33%)
Cliente 1: INADIMPLENTE (risco de default: 51.63%)
Cliente 2: ADIMPLENTE (risco de default: 6.57%)
```

Para inferir sobre clientes reais, basta passar um `DataFrame` com as mesmas colunas do CSV original (exceto `ID` e o target) para `inferir(dados_novos, nome_modelo='classificador_rf')`.

---

## Atendimento aos critérios de avaliação

| Critério                       | Como é atendido                                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Pipeline completo               | `main.py`orquestra carregamento → preparação → split → preprocessador → SMOTE → treinamento → avaliação |
| Seleção do melhor modelo      | Comparação por acurácia entre RF e KNN, com print do vencedor                                                    |
| Módulo de inferência          | `inferir.py`carrega os `.pkl`salvos e prediz sem retreinar                                                      |
| Distribuição de probabilidade | `predict_proba`retorna `prob_nao_default`e `prob_default`                                                     |

---

## Dataset

*Default of Credit Card Clients* (UCI Machine Learning Repository): 30.000 clientes de cartão de crédito em Taiwan (abril a setembro/2005), com 23 atributos preditivos e 1 alvo binário. Aproximadamente 78% adimplentes (classe 0) e 22% inadimplentes (classe 1).
