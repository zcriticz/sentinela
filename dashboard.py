import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from data_generator import generate_historical_data
from ml_model import run_full_analysis

st.set_page_config(page_title="Sentinela", page_icon="🧑🏼‍🚒", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@400;600;700&display=swap');
    
    .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
    
    h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; color: #e94560 !important; letter-spacing: 2px; }
    
    .stMetric {
        background: linear-gradient(145deg, #1f4068 0%, #162447 100%);
        padding: 20px; border-radius: 15px; border-left: 4px solid #e94560;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .stMetric label { color: #a8a8a8 !important; font-family: 'Montserrat', sans-serif; font-weight: 600; }
    .stMetric [data-testid="stMetricValue"] { color: #00d9ff !important; font-family: 'Bebas Neue', sans-serif; font-size: 2.5rem !important; }
    
    .insight-box {
        background: linear-gradient(145deg, #1f4068 0%, #162447 100%);
        padding: 15px 20px; border-radius: 12px; margin: 10px 0; border-left: 4px solid #ffc107;
        color: #e8e8e8; font-family: 'Montserrat', sans-serif; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .cluster-card {
        background: linear-gradient(145deg, #2d4a6e 0%, #1f3756 100%);
        padding: 15px; border-radius: 10px; margin: 5px; border-top: 3px solid #e94560;
    }
    
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #162447 0%, #1f4068 100%); }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stMultiSelect label,
    div[data-testid="stSidebar"] p { color: #e8e8e8 !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: transparent; border-bottom: 2px solid #1f4068; padding-bottom: 0; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f4068; border-radius: 10px 10px 0 0; color: #e8e8e8 !important;
        font-family: 'Montserrat', sans-serif; font-weight: 600; font-size: 14px;
        padding: 12px 24px; border: none; transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #2d4a6e; color: white !important; }
    .stTabs [aria-selected="true"] { background-color: #e94560 !important; color: white !important; border-bottom: 3px solid #e94560; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

COLORS = {'primary': '#e94560', 'secondary': '#00d9ff', 'background': '#162447', 'text': '#e8e8e8', 'accent': '#ffc107', 'success': '#00c853', 'danger': '#ff5252'}

TYPE_COLORS = {
    'Incêndio': '#e94560', 'Pré-Hospitalar': '#00d9ff', 'Salvamento': '#ffc107',
    'Produtos Perigosos': '#9c27b0', 'Prevenção': '#00c853', 'Atividade Comunitária': '#ff9800'
}


@st.cache_data
def load_data():
    return generate_historical_data()


@st.cache_data
def run_ml(_df):
    return run_full_analysis(_df)


def create_donut_chart(df, column, title):
    counts = df[column].value_counts()
    colors = [TYPE_COLORS.get(t, '#666666') for t in counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values, hole=0.5, marker_colors=colors,
        textinfo='label+percent', textposition='outside', textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{label}</b><br>%{value}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')), showlegend=True,
        legend=dict(font=dict(color='white', size=11), bgcolor='rgba(0,0,0,0.3)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=60, b=20, l=20, r=20), height=400
    )
    return fig


def create_timeline_chart(df, title):
    df_temp = df.copy()
    df_temp['year_month'] = pd.to_datetime(df_temp['datetime']).dt.to_period('M').astype(str)
    monthly = df_temp.groupby(['year_month', 'type']).size().reset_index(name='count')
    
    fig = px.line(monthly, x='year_month', y='count', color='type', color_discrete_map=TYPE_COLORS, markers=True)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')),
        xaxis_title='', yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'), title_font=dict(color='white')),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'), title_font=dict(color='white')),
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.3)'), height=400
    )
    return fig


def create_histogram(df, column, title, color=COLORS['primary']):
    fig = px.histogram(df, x=column, nbins=20, color_discrete_sequence=[color])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')),
        xaxis_title='', yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'), title_font=dict(color='white')),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'), title_font=dict(color='white')),
        height=400
    )
    return fig


def create_boxplot(df, x_col, y_col, title):
    fig = px.box(df, x=x_col, y=y_col, color=x_col, color_discrete_map=TYPE_COLORS)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')),
        xaxis_title='', yaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white', size=10), tickangle=45),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'), title_font=dict(color='white')),
        showlegend=False, height=400
    )
    return fig


def create_horizontal_bar(df, x_col, y_col, title, color=COLORS['secondary']):
    data = df.groupby(y_col)[x_col].count().sort_values(ascending=True).tail(10)
    
    fig = go.Figure(go.Bar(
        x=data.values, y=data.index, orientation='h', marker_color=color,
        text=data.values, textposition='outside', textfont=dict(color='white')
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')), xaxis_title='',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'), title_font=dict(color='white')),
        yaxis=dict(tickfont=dict(color='white')), height=400
    )
    return fig


def create_map(df, sample_size=500):
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    center_lat, center_lon = sample_df['latitude'].mean(), sample_df['longitude'].mean()
    
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB dark_matter')
    
    marker_colors = {
        'Incêndio': 'red', 'Pré-Hospitalar': 'blue', 'Salvamento': 'orange',
        'Produtos Perigosos': 'purple', 'Prevenção': 'green', 'Atividade Comunitária': 'cadetblue'
    }
    
    for _, row in sample_df.iterrows():
        color = marker_colors.get(row['type'], 'gray')
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="color: {TYPE_COLORS.get(row['type'], '#666')}; margin: 0;">{row['type']}</h4>
            <p style="margin: 5px 0;"><b>Subtipo:</b> {row['subtype']}</p>
            <p style="margin: 5px 0;"><b>Bairro:</b> {row['neighborhood']}</p>
            <p style="margin: 5px 0;"><b>Data:</b> {row['date']}</p>
            <p style="margin: 5px 0;"><b>Gravidade:</b> {row['severity']}/5</p>
        </div>
        """
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']], radius=6,
            popup=folium.Popup(popup_html, max_width=250),
            color=color, fill=True, fill_color=color, fill_opacity=0.7
        ).add_to(map_obj)
    
    return map_obj


def create_gauge(value, title, max_value=100, color=COLORS['primary']):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        number={'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickfont': {'color': 'white'}},
            'bar': {'color': color}, 'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 2, 'bordercolor': 'rgba(255,255,255,0.3)',
            'steps': [
                {'range': [0, max_value*0.5], 'color': 'rgba(0,200,83,0.3)'},
                {'range': [max_value*0.5, max_value*0.75], 'color': 'rgba(255,193,7,0.3)'},
                {'range': [max_value*0.75, max_value], 'color': 'rgba(233,69,96,0.3)'}
            ]
        }
    ))
    
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'}, height=250, margin=dict(t=50, b=20, l=30, r=30))
    return fig


def main():
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3rem; margin: 0;">SENTINELA</h1>
        <h3 style="color: #a8a8a8 !important; font-family: 'Montserrat', sans-serif; font-weight: 400; letter-spacing: 1px;">
            Corpo de Bombeiros - Região Metropolitana do Recife
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_data()
    
    with st.sidebar:
        st.markdown("## FILTROS")
        years = sorted(df['datetime'].dt.year.unique())
        selected_years = st.multiselect("Ano", options=years, default=years)
        
        types = df['type'].unique().tolist()
        selected_types = st.multiselect("Tipo", options=types, default=types)
        
        neighborhoods = df['neighborhood'].unique().tolist()
        selected_neighborhoods = st.multiselect("Bairro", options=neighborhoods, default=neighborhoods)
        
        severity_range = st.slider("Gravidade", min_value=1, max_value=5, value=(1, 5))
    
    filtered_df = df[
        (df['datetime'].dt.year.isin(selected_years)) &
        (df['type'].isin(selected_types)) &
        (df['neighborhood'].isin(selected_neighborhoods)) &
        (df['severity'] >= severity_range[0]) &
        (df['severity'] <= severity_range[1])
    ]
    
    if len(filtered_df) == 0:
        st.warning("Nenhum dado encontrado.")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total", f"{len(filtered_df):,}")
    with col2:
        avg_response = filtered_df['response_time_min'].mean()
        st.metric("Tempo Médio", f"{avg_response:.1f} min")
    with col3:
        total_victims = filtered_df['victims'].sum()
        st.metric("Vítimas", f"{total_victims:,}")
    with col4:
        avg_severity = filtered_df['severity'].mean()
        st.metric("Gravidade Média", f"{avg_severity:.2f}")
    with col5:
        resolution_rate = (filtered_df['outcome'] == 'Resolvido no Local').mean() * 100
        st.metric("Resolução Local", f"{resolution_rate:.1f}%")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Análise Espacial", "Machine Learning", "Dados"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_donut_chart(filtered_df, 'type', 'Distribuição por Tipo'), use_container_width=True)
        with col2:
            st.plotly_chart(create_donut_chart(filtered_df, 'shift', 'Distribuição por Turno'), use_container_width=True)
        
        st.plotly_chart(create_timeline_chart(filtered_df, 'Evolução Temporal'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_histogram(filtered_df, 'response_time_min', 'Tempo de Resposta (min)', COLORS['secondary']), use_container_width=True)
        with col2:
            st.plotly_chart(create_boxplot(filtered_df, 'type', 'service_time_min', 'Tempo de Atendimento por Tipo'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_horizontal_bar(filtered_df, 'id', 'neighborhood', 'Top 10 Bairros', COLORS['primary']), use_container_width=True)
        with col2:
            st.plotly_chart(create_horizontal_bar(filtered_df, 'id', 'subtype', 'Top 10 Subtipos', COLORS['accent']), use_container_width=True)
    
    with tab2:
        st_folium(create_map(filtered_df), width=None, height=500, use_container_width=True)
        
        pivot = filtered_df.pivot_table(index='neighborhood', columns='type', values='id', aggfunc='count', fill_value=0)
        fig_heatmap = px.imshow(pivot, color_continuous_scale='YlOrRd', aspect='auto')
        fig_heatmap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(color='white'), tickangle=45),
            yaxis=dict(tickfont=dict(color='white')),
            coloraxis_colorbar=dict(title=dict(text='Qtd', font=dict(color='white')), tickfont=dict(color='white')),
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        ml_results = run_ml(filtered_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            accuracy = ml_results['severity_model']['accuracy'] * 100
            st.plotly_chart(create_gauge(accuracy, "Acurácia (%)", 100, COLORS['success']), use_container_width=True)
        with col2:
            mae = ml_results['response_model']['mae']
            st.plotly_chart(create_gauge(mae, "MAE (min)", 30, COLORS['accent']), use_container_width=True)
        with col3:
            silhouette = ml_results['clustering']['silhouette_score'] * 100
            st.plotly_chart(create_gauge(silhouette, "Silhouette (%)", 100, COLORS['secondary']), use_container_width=True)
        
        importance = ml_results['severity_model']['feature_importance']
        df_importance = pd.DataFrame({'feature': importance.keys(), 'importance': importance.values()}).sort_values('importance', ascending=True)
        
        fig_importance = go.Figure(go.Bar(
            x=df_importance['importance'], y=df_importance['feature'], orientation='h',
            marker_color=COLORS['secondary'], text=[f'{v:.1%}' for v in df_importance['importance']],
            textposition='outside', textfont=dict(color='white')
        ))
        fig_importance.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='white'), tickformat='.0%'),
            yaxis=dict(tickfont=dict(color='white')), height=300
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        cluster_analysis = ml_results['clustering']['cluster_analysis']
        cols = st.columns(len(cluster_analysis))
        for i, (name, info) in enumerate(cluster_analysis.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="cluster-card">
                    <h4 style="color: {COLORS['primary']}; margin: 0;">{name}</h4>
                    <p style="color: white; margin: 5px 0;">
                        <b>Tipo:</b> {info['main_type']}<br>
                        <b>Turno:</b> {info['main_shift']}<br>
                        <b>Gravidade:</b> {info['avg_severity']:.2f}<br>
                        <b>Tempo:</b> {info['avg_response_time']:.1f} min<br>
                        <b>N:</b> {info['incident_count']} ({info['percentage']:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        for insight in ml_results['insights']:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        forecast = ml_results['demand_forecast']
        trend = forecast['trend']
        rate = forecast['monthly_rate']
        
        if trend == 'increasing':
            st.success(f"Tendência CRESCENTE: +{rate:.1f} ocorrências/mês")
        elif trend == 'decreasing':
            st.info(f"Tendência DECRESCENTE: {rate:.1f} ocorrências/mês")
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(filtered_df[['response_time_min', 'service_time_min', 'severity', 'victims']].describe().round(2), use_container_width=True)
        with col2:
            st.dataframe(filtered_df['outcome'].value_counts(), use_container_width=True)
        
        display_cols = ['id', 'datetime', 'type', 'subtype', 'neighborhood', 'severity', 'response_time_min', 'victims', 'outcome']
        st.dataframe(filtered_df[display_cols].sort_values('datetime', ascending=False), use_container_width=True, height=400)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name="incidents.csv", mime="text/csv")


if __name__ == '__main__':
    main()
