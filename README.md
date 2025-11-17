# Sistema Inteligente de Prevenção de Acidentes Rodoviários

**Status:** Dashboard Funcional e Modelo de ML Integrado (v1.0)

Este projeto implementa um Dashboard Analítico construído com Dash/Plotly para analisar padrões de acidentes e utilizar um modelo de Machine Learning para prever o nível de risco (ALTO, MÉDIO, BAIXO) em tempo real, com base na hora do dia e na causa do acidente.

---

## 1. Estrutura do Projeto

| Pasta/Arquivo | Conteúdo | Finalidade |
| :--- | :--- | :--- |
| `src/dashboard.py` | Código principal do Dash | Execução do aplicativo e callbacks de previsão. |
| `src/train_model.py` | Script de Treinamento | Gera o modelo e os codificadores (`.pkl`). |
| `data/processed/` | `df_final_ml.csv` | Dataset limpo e processado (essencial para gráficos e tabelas). |
| `models/` | Arquivos `.pkl` | Modelos de Machine Learning e codificadores prontos para uso. |
| `assets/` | `style.css` | Estilos CSS personalizados e tema visual do painel. |
| `requirements.txt` | Lista de dependências | Usado para instalar o ambiente Python. |

---

## 2. Guia de Instalação e Execução

Siga estes passos exatos para rodar o dashboard localmente:

### Pré-requisitos

* Python 3.8+ instalado.
* Conexão à internet (para carregar o mapa).

### Passo 1: Clone o Repositório

```bash
git clone https://github.com/Pablohdantas/sistema_prevencao_acidentes
cd sistema_prevencao_acidentes
```
### Passo 2: Criar ambiente de desenvolvimento
```bash
# Criar o ambiente
python3 -m venv .venv

# Ativar (macOS/Linux)
source .venv/bin/activate

# Ativar (Windows)
.\.venv\Scripts\activate
```
### Passo 3: Instalar dependências
```bash
pip install -r requirements.txt
```
### Passo 4: Download do Dataset Essencial

O arquivo principal de dados (`df_final_ml.csv`) excede o limite de 100 MB do GitHub.

**Atenção:** O dashboard **não funcionará** sem este arquivo.

1.  **Baixe o arquivo**:
    [dt_final_ml.csv](https://drive.google.com/file/d/1gyYByUQebV5riYrrnP9ANqd-KtSHMHFt/view?usp=drive_link)
    
2.  Após o download, crie a pasta **`data`** na raiz do projeto dentro dela cria a pasta **`processed`** e mova o arquivo **`df_final_ml.csv`** para:

    `sistema_prevencao_acidentes/data/processed/`


### Passo 5: Execute o dashboard
```bash
python src/dashboard.py
```
## 3. Documentação Técnica
* **Análise Exploratória (EDA):** [Notebook de EDA](https://github.com/Pablohdantas/sistema_prevencao_acidentes/blob/main/notebooks/eda.ipynb)
