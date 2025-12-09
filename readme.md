# Sentinela - Painel Analítico para Corpo de Bombeiros RM Recife

Bem-vindo ao **Sentinela**, um projeto direcionado à Faculdade Senac para análise de dados desenvolvido para monitoramento e visualização das ocorrências atendidas pelo Corpo de Bombeiros na Região Metropolitana do Recife! Este projeto utiliza tecnologias modernas de Data Science e Visualização Interativa para fornecer insights rápidos, inteligentes e amigáveis sobre os dados operacionais da corporação.


## Funcionalidades

- **Visualização de Indicadores Chave**  
  Veja instantaneamente totais de ocorrências, tempo médio de resposta, gravidade, taxas de resolução e muito mais.

- **Filtros Interativos Avançados**  
  Filtre por ano, tipo de ocorrência, bairro, gravidade; os dashboards se atualizam automaticamente.

- **Dashboards Visuais**  
  - Gráficos Rosquinha, Linha Temporal, Barras, Histogramas, Boxplots dinâmicos (Plotly)
  - Análise espacial: mapas interativos, clusters e mapa de calor por bairro/tipo (Folium & Plotly)
  
- **Machine Learning Integrado**  
  - Predição de gravidade e tempo de atendimento
  - Clustering de ocorrências
  - Visualização de importância das variáveis dos modelos

- **Simulador de Dados**  
  Dados sintéticos gerados realistas cobrindo diferentes tipos de ocorrências, bairros, subtipos e turnos.

## Tecnologias

| Tecnologia         | Descrição                    | Link                                                      |
|--------------------|-----------------------------|-----------------------------------------------------------|
| Streamlit          | Web app interativo           | [streamlit.io](https://streamlit.io/)                     |
| Pandas             | Manipulação de dados         | [pandas.pydata.org](https://pandas.pydata.org/)           |
| Numpy              | Operações numéricas          | [numpy.org](https://numpy.org/)                           |
| Plotly             | Gráficos interativos         | [plotly.com/python/](https://plotly.com/python/)          |
| Folium             | Mapas/visualização geoesp.   | [python-visualization.github.io/folium/](https://python-visualization.github.io/folium/) |
| Scikit-learn       | Machine learning             | [scikit-learn.org](https://scikit-learn.org/)             |
| streamlit-folium   | Folium em Streamlit          | [github.com/randyzwitch/streamlit-folium](https://github.com/randyzwitch/streamlit-folium) |

## Iniciando Localmente

- 1. Crie um ambiente virtual (opcional mas recomendado)

```
python3 -m venv .venv
source .venv/bin/activate  # (ou .venv\Scripts\activate no Windows)
```

- 2. Instale as dependências usando o comando
``pip install -r requirements.txt``

- 3. Rode o painel Streamlit com
``streamlit run dashboard.py``

Os dados são simulados automaticamente na inicialização, não há necessidade de baixar datasets.

## Principais Dashboards

- **Indicadores Gerais:** Total de ocorrências, vítimas, tempo médio, gravidade, taxa de resolução local.
- **Séries Temporais:** Evolução mensal/diária das ocorrências.
- **Top Bairros/Publico/Tipos:** Ranking dos bairros e ocorrências mais frequentes.
- **Mapas Interativos:** Ocorrências geolocalizadas + mapa de calor.
- **Modelos de ML:** Métricas, predições e insights.
- **Tabela de Dados:** Consulta e exportação dos dados sintéticos filtrados.

## Contribuições

Fique à vontade para abrir issues ou PRs! Sugestões, correções e novas ideias são bem-vindas para evoluir o painel.

## Observações

- Por se tratar de um projeto acadêmico, o painel utiliza **dados sintéticos**, não reais.
- Projeto aberto para fins educacionais, inspirando-se em práticas modernas de análise e visualização de dados.


## Autor
- [Cristian Santos](https://github.com/zcriticz)

## Licença
Esse projeto é licenciado através do MIT. Para mais informações, leia [license](license)


