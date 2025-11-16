import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


# ---------------------------------------------------------
# CONFIGURAÇÕES E CONSTANTES
# ---------------------------------------------------------
DF_PATH = 'data/processed/df_final_ml.csv'
MODEL_DIR = 'models'

RISCO_ALTO = ['FALTA DE ATENÇÃO', 'INGESTÃO DE ÁLCOOL', 'DORMIR AO VOLANTE']
RISCO_MEDIO = ['VELOCIDADE INCOMPATÍVEL', 'DEFEITO MECÂNICO']


# ---------------------------------------------------------
# FUNÇÃO: Classificação de risco
# ---------------------------------------------------------
def classify_risk(causa):
    if pd.isna(causa):
        return 'BAIXO'
    causa_upper = str(causa).strip().upper()
    if causa_upper in RISCO_ALTO:
        return 'ALTO'
    elif causa_upper in RISCO_MEDIO:
        return 'MÉDIO'
    return 'BAIXO'


# ---------------------------------------------------------
# 1. CARREGAMENTO DO DATASET
# ---------------------------------------------------------
print("\n=== CARREGANDO BASE ===")

try:
    df = pd.read_csv(DF_PATH, low_memory=False)
    print(f"✔ Dados carregados de: {DF_PATH}")
except FileNotFoundError:
    print(f"Arquivo não encontrado. Usando dados mock.")
    df = pd.DataFrame({
        'hora_do_dia': np.random.randint(0, 24, 1000),
        'causa_acidente': np.random.choice(
            ['FALTA DE ATENÇÃO', 'VELOCIDADE INCOMPATÍVEL', 'CHUVA',
             'DORMIR AO VOLANTE', 'OUTROS'], 1000)
    })


# ---------------------------------------------------------
# 2. CRIAÇÃO DO TARGET
# ---------------------------------------------------------
print("\n=== CRIANDO TARGET 'risco_label' ===")
df['risco_label'] = df['causa_acidente'].apply(classify_risk)
print("✔ Target criado.")


# ---------------------------------------------------------
# 3. ENGENHARIA DE FEATURES
# ---------------------------------------------------------
print("\n=== ENGENHARIA DE FEATURES ===")

# Limpeza de string
df['causa_limpa'] = df['causa_acidente'].astype(str).str.strip().str.upper()

# Codificador FEATURE
le_causa = LabelEncoder()
df['causa_encoded'] = le_causa.fit_transform(df['causa_limpa'])

# Codificador TARGET
le_risco = LabelEncoder()
df['risco_target'] = le_risco.fit_transform(df['risco_label'])

FEATURES = ['hora_do_dia', 'causa_encoded']
TARGET = 'risco_target'

X = df[FEATURES]
y = df[TARGET]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✔ Codificação concluída.")


# ---------------------------------------------------------
# 4. TREINAMENTO DO MODELO
# ---------------------------------------------------------
print("\n=== TREINANDO MODELO RANDOM FOREST ===")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("✔ Treinamento concluído.")

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)


# ---------------------------------------------------------
# 5. AVALIAÇÃO
# ---------------------------------------------------------
print("\n=== AVALIAÇÃO DO MODELO ===")
print(f"Acurácia: {accuracy:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(
    y_test, y_pred,
    target_names=le_risco.classes_
))


# ---------------------------------------------------------
# 6. SALVAR MODELOS E CODIFICADORES
# ---------------------------------------------------------
print("\n=== SALVANDO ARQUIVOS ===")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, 'risk_predictor_model.pkl'))
joblib.dump(le_risco, os.path.join(MODEL_DIR, 'le_risco.pkl'))
joblib.dump(le_causa, os.path.join(MODEL_DIR, 'le_causa.pkl'))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, 'features.pkl'))

print(f"✔ Modelos salvos em: {MODEL_DIR}")
print("✔ Arquivos gerados: model, le_risco, le_causa, features")
print("\n🏁 Processo finalizado com sucesso!")
