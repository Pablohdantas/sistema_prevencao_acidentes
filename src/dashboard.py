import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
import numpy as np
import joblib 
from dash.dependencies import Input, Output
import os

PLOTLY_TEMPLATE = 'plotly_white' 

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'risk_predictor_model.pkl')
LE_RISCO_PATH = os.path.join(MODEL_DIR, 'le_risco.pkl')
LE_CAUSA_PATH = os.path.join(MODEL_DIR, 'le_causa.pkl')

try:
    MODEL_FEATURES = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))
except FileNotFoundError:
    MODEL_FEATURES = ['hora_do_dia', 'causa_encoded']


def create_accident_map(df_final):
    df_map = df_final.copy()
    
    df_map['latitude'] = pd.to_numeric(df_map['latitude'].astype(str).replace(",", ".", regex=True), errors='coerce')
    df_map['longitude'] = pd.to_numeric(df_map['longitude'].astype(str).replace(",", ".", regex=True), errors='coerce')
    df_map.dropna(subset=['latitude', 'longitude'], inplace=True)

    if len(df_map) == 0:
        return go.Figure()

    lat = df_map['latitude'].values
    lon = df_map['longitude'].values
    bins = 80

    heatmap, lat_edges, lon_edges = np.histogram2d(lat, lon, bins=bins)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

    z = heatmap.flatten()
    lats_flat = np.repeat(lat_centers, bins)
    lons_flat = np.tile(lon_centers, bins)

    mask = z > 0
    z = z[mask]
    lats_flat = lats_flat[mask]
    lons_flat = lons_flat[mask]

    fig = go.Figure(go.Densitymap(
        lat=lats_flat,
        lon=lons_flat,
        z=z,
        radius=18,
        colorscale="Hot",
        showscale=True,
    ))

    fig.update_layout(
        mapbox_style="open-street-map", 
        mapbox_center={"lat": -15, "lon": -48},
        mapbox_zoom=3,
        height=500,
        margin=dict(r=0, t=0, l=0, b=0),
        template=PLOTLY_TEMPLATE
    )
    return fig


DATA_PATH = 'data/processed/df_final_ml.csv'

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("Dashboard: Dados carregados.")
except FileNotFoundError:
    print(f"ERRO: Dataset não encontrado em {DATA_PATH}. Usando mock.")
    df = pd.DataFrame({
        "hora_do_dia": [18,19,17,10,11], 
        "causa_acidente": ["ATENÇÃO", "VELOCIDADE","ATENÇÃO","CHUVA","SONO"],
        "latitude": [-15.01, -15.01, -23.15],
        "longitude": [-48.01, -48.01, -46.50]
    })


df["hora_do_dia"] = pd.to_numeric(df["hora_do_dia"], errors="coerce").fillna(0)

risco_por_hora = df["hora_do_dia"].value_counts().sort_index().reset_index()
risco_por_hora.columns = ["Hora", "Contagem de Acidentes"]
risco_por_hora["Hora"] = risco_por_hora["Hora"].astype(str)

causas_principais = df["causa_acidente"].value_counts().head(5).reset_index()
causas_principais.columns = ["Causa", "Contagem"]

try:
    ML_MODEL = joblib.load(MODEL_PATH)
    LE_RISCO = joblib.load(LE_RISCO_PATH)
    LE_CAUSA = joblib.load(LE_CAUSA_PATH)
    print("ML: Modelo (2 Features) e codificadores carregados com sucesso.")
except FileNotFoundError:
    print("ML: Erro ao carregar o modelo PKL. Retornando ao MOCK.")
    ML_MODEL = None


def prever_risco(modelo, hora, causa):
    cores_risco = {"ALTO": "#B91C1C", "MÉDIO": "#F59E0B", "BAIXO": "#059669", "ERRO": "#4B5563"}
    
    if modelo is None:
        if 17 <= hora <= 20 and causa == causas_principais["Causa"].iloc[0]:
            risco = "ALTO"
            local = "BR-116/SP (Mock)"
        else:
            risco = "BAIXO"
            local = "BR-101/RJ (Mock)"
        return risco, cores_risco[risco], local

    try:
        causa_limpa = str(causa).strip().upper() 
        
        try:
            causa_encoded = LE_CAUSA.transform([causa_limpa])[0] 
        except ValueError:
            print(f"Aviso ML: Causa '{causa_limpa}' não vista no treino. Usando fallback (0).")
            causa_encoded = 0 
        
        input_data = pd.DataFrame([[hora, causa_encoded]], columns=MODEL_FEATURES)
        
        pred_encoded = modelo.predict(input_data)[0]
        risco = LE_RISCO.inverse_transform([pred_encoded])[0]
        
        local = f"ML Preditor (Hora {hora}h)" 
        return risco, cores_risco[risco], local

    except Exception as e:
        print(f"Erro na previsão ML: {e}")
        return "ERRO", cores_risco.get("ERRO"), "Verificar Logs"


def top_50_trechos(df):
    dfc = df.copy()
    
    dfc["latitude"] = pd.to_numeric(dfc["latitude"].astype(str).replace(",", ".", regex=True), errors='coerce')
    dfc["longitude"] = pd.to_numeric(dfc["longitude"].astype(str).replace(",", ".", regex=True), errors='coerce')
    dfc.dropna(subset=["latitude", "longitude"], inplace=True)

    dfc["lat_round"] = dfc["latitude"].round(4)
    dfc["lon_round"] = dfc["longitude"].round(4)

    ranking = (
        dfc.groupby(["lat_round", "lon_round"])
        .size()
        .reset_index(name="acidentes")
        .sort_values("acidentes", ascending=False)
        .head(50)
    )

    return ranking


def recomendar_intervencao(qtd):
    if qtd >= 40: return "Instalar lombada + iluminação 💡🚧"
    if qtd >= 30: return "Melhorar sinalização horizontal 🟡"
    if qtd >= 20: return "Recapeamento + pintura de faixas 🎨"
    return "Monitoramento periódico 🔍"


ranking_50 = top_50_trechos(df)
ranking_50["Recomendação"] = ranking_50["acidentes"].apply(recomendar_intervencao)


app = Dash(__name__) 

causas_limpas_unicas = df['causa_acidente'].astype(str).str.strip().str.upper().unique()
causas_options = [{'label': causa, 'value': causa} for causa in causas_limpas_unicas]
default_causa = str(causas_principais["Causa"].iloc[0]).strip().upper()

app.layout = html.Div(className='main-wrapper', id='app-container', children=[
    
    html.Div(className='container', children=[
        html.Hgroup(
            children=[
                html.H1("Sistema Inteligente de Prevenção de Acidentes"),
                html.H2("Análise Preditiva + Top 50 Trechos Críticos"),
            ], style={'textAlign': 'center', 'marginBottom': '30px'}
        ),
    ]),

    html.Section(className='grid container', children=[
        html.Article(className='col', children=[
            html.H3("1. Risco por Hora (Clique para Prever)"),
            dcc.Graph(
                id='risco-por-hora',
                figure=px.bar(
                    risco_por_hora, x="Hora", y="Contagem de Acidentes",
                    color_discrete_sequence=["#2563EB"],
                    template=PLOTLY_TEMPLATE
                ).update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0)), 
                style={'height':'400px'}
            )
        ]),

        html.Article(className='col', children=[
            html.H3("2. Principais Causas"),
            dcc.Graph(
                id='top-causas',
                figure=px.bar(
                    causas_principais, y="Causa", x="Contagem",
                    orientation="h", color_discrete_sequence=["#F59E0B"],
                    template=PLOTLY_TEMPLATE
                ).update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0)),
                style={'height':'400px'}
            )
        ])
    ]),

    html.Section(className='grid container', children=[

        html.Article(className='col-7', children=[
            html.H3("3. Mapa de Calor dos Acidentes"),
            dcc.Graph(
                id="mapa",
                figure=create_accident_map(df),
                style={'height':'500px'}
            )
        ]),

        html.Article(className='col-3', children=[
            html.H3("4. Previsão de Risco (ML/PLN)"),
            
            html.Label("Selecione a Causa (PLN):"),
            dcc.Dropdown(
                id='causa-dropdown',
                options=causas_options,
                value=default_causa,
                clearable=False
            ),
            html.Hr(),
            
            html.Div(id='risco-previsto-texto',
                    children="Clique em uma hora!",
                    style={'textAlign':'center'}),

            html.Div(id='risco-previsto-valor',
                    children="BAIXO",
                    style={'textAlign':'center','fontSize':'25px','fontWeight':'bold','paddingBottom': '15px'}),
            
            html.H4("Chat de Intervenção"),
            html.P("Simulação de interação de chat e alertas de rotina.", style={'fontSize': '0.9em'}),
        ])
    ]),

    html.Article(className='container', children=[
        
        html.H3("5. Top 50 Trechos Críticos com Ações Recomendadas"),

        dcc.Graph(
            id="ranking-50",
            figure=px.scatter_map( 
                ranking_50,
                lat="lat_round",
                lon="lon_round",
                size="acidentes",
                color="acidentes",
                hover_data={"acidentes": True, "Recomendação": True, "lat_round": False, "lon_round": False},
                zoom=3,
                height=500,
                color_continuous_scale=px.colors.sequential.Plasma,
                template=PLOTLY_TEMPLATE
            ),
            style={'height':'500px'}
        ),

        html.H4("Tabela dos 50 Trechos:"),

       dash_table.DataTable(
            id="tabela-ranking",
            columns=[
                {"name": "Latitude", "id": "lat_round"},
                {"name": "Longitude", "id": "lon_round"},
                {"name": "Acidentes", "id": "acidentes"},
                {"name": "Recomendação", "id": "Recomendação"}
            ],
            data=ranking_50.to_dict("records"),
            page_size=10,
            sort_action="native",
            filter_action="native",

            style_table={'height': '400px', 'overflowY': 'auto'},
            style_header={'backgroundColor': '#1E3A8A', 'color': 'white', 'fontWeight': 'bold'},
            style_cell={'padding': '10px', 'fontFamily': 'Arial', 'fontSize': '13.5px', 'textAlign': 'left'},
            style_data={'border': 'none'},
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': '#F9FAFB'},
                {'if': {'state': 'selected'}, 'backgroundColor': '#2563EB', 'color': 'white'},
                {'if': {'column_id': 'Recomendação'}, 'fontWeight': 'bold', 'color': '#1E3A8A'},
                {'if': {'filter_query': '{acidentes} >= 40', 'column_id': 'acidentes'}, 'color': '#B91C1C', 'fontWeight': 'bold'},
                {'if': {'filter_query': '{acidentes} >= 30 && {acidentes} < 40', 'column_id': 'acidentes'}, 'color': '#F59E0B', 'fontWeight': 'bold'},
                {'if': {'filter_query': '{acidentes} < 30', 'column_id': 'acidentes'}, 'color': '#059669', 'fontWeight': 'bold'},
            ]
        )
    ])
])


@app.callback(
    Output('risco-previsto-texto', 'children'),
    Output('risco-previsto-valor', 'children'),
    Output('risco-previsto-valor', 'style'),
    Input('risco-por-hora', 'clickData'),
    Input('causa-dropdown', 'value')
)
def update_previsao(clickData, causa_selecionada):
    
    cores_risco = {"ALTO": "#B91C1C", "MÉDIO": "#F59E0B", "BAIXO": "#059669", "ERRO": "#4B5563"}

    if clickData is None:
        hora = 18
        status_texto = "Risco Previsto (Padrão 18h)"
    else:
        hora = int(clickData["points"][0]["x"])
        status_texto = f"Risco Previsto às {hora}h"

    if causa_selecionada is None:
        causa_selecionada = default_causa 
    risco, cor, local = prever_risco(ML_MODEL, hora, causa_selecionada)
    
    return f"{status_texto} — Causa: {causa_selecionada}", risco, {"color":cor,"fontWeight":"bold", 'textAlign':'center','fontSize':'25px'}


if __name__ == "__main__":
    app.run(debug=True)
