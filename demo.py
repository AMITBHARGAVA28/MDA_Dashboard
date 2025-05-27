# -*- coding: utf-8 -*-
"""Horizon Europe Dashboard PRO - Fixed Version"""

# ====================
# CONFIGURATION (MUST BE AT TOP)
# ====================
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Horizon Europe Dashboard PRO",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objs as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import io
from pyvis.network import Network
import altair as alt
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from itertools import combinations
import tempfile
import warnings
import gc
warnings.filterwarnings('ignore')

# ====================
# FIXED CSS
# ====================
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    h1 {
        color: #003366;
        border-bottom: 2px solid #003366;
        padding-bottom: 10px;
    }
    .stMetric {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .stMetric:hover {
        transform: translateY(-5px);
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .network-container {
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ====================
# DATA PREPARATION
# ====================
@st.cache_data(ttl=3600)
def load_data():
    # Try different encodings - start with the most likely
    encodings = ['latin1', 'ISO-8859-1', 'utf-16', 'cp1252', 'utf-8']
    
    for encoding in encodings:
        try:
            org_df = pd.read_csv("horizon_organizations.csv", encoding=encoding)
            proj_df = pd.read_csv("horizon_projects.csv", encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not read files with any of the tried encodings")
    
    # Date processing with correct format
    proj_df['startDate'] = pd.to_datetime(proj_df['startDate'], format='%d-%m-%Y', errors='coerce')
    proj_df['endDate'] = pd.to_datetime(proj_df['endDate'], format='%d-%m-%Y', errors='coerce')
    proj_df['start_year'] = proj_df['startDate'].dt.year
    
    # Enhanced funding categories
    funding_scheme_groups = {
        "ERC": ["ERC", "HORIZON-ERC", "HORIZON-ERC-SYG", "ERC-POC"],
        "MSCA": ["MSCA", "HORIZON-TMA-MSCA", "MSCA-IF", "MSCA-COFUND"],
        "EIC": ["EIC", "HORIZON-EIC", "EIC-ACC"],
        "RIA/IA": ["RIA", "IA", "HORIZON-RIA", "HORIZON-IA"],
        "CSA": ["CSA", "HORIZON-CSA", "HORIZON-JU-CSA"],
        "COFUND": ["COFUND", "HORIZON-COFUND", "ERA-NET-Cofund"],
        "Joint Undertakings": ["HORIZON-JU", "HORIZON-JU-RIA"]
    }
    
    proj_df['fundingCategory'] = "Other"
    for category, schemes in funding_scheme_groups.items():
        for scheme in schemes:
            proj_df.loc[proj_df['fundingScheme'].str.contains(scheme, na=False), 'fundingCategory'] = category
    
    # Topic clustering
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(proj_df['cleaned_text'].fillna(''))
    proj_df['topic_cluster'] = KMeans(n_clusters=8).fit_predict(X)
    
    return org_df, proj_df

org_df, proj_df = load_data()

# ====================
# UTILITY FUNCTIONS
# ====================
def format_currency(value):
    if value >= 1e9:
        return f"€{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"€{value/1e6:.1f}M"
    else:
        return f"€{value/1e3:,.0f}K"

def create_network_graph(org_df, min_projects=3):
    # Create collaboration network
    project_orgs = org_df.groupby('projectID')['organisationID'].apply(set)
    edge_counter = defaultdict(int)
    
    for orgs in project_orgs:
        for u, v in combinations(sorted(orgs), 2):
            edge_counter[(u, v)] += 1
    
    G = nx.Graph()
    G.add_edges_from((u, v, {'weight': w}) for (u, v), w in edge_counter.items())
    
    # Filter nodes
    degrees = dict(G.degree())
    filtered_nodes = [n for n, d in degrees.items() if d >= min_projects]
    G = G.subgraph(filtered_nodes)
    
    return G

def run_scenario(data, adjustment_factor, years):
    scenario_data = data.copy()
    scenario_data['adjusted_funding'] = scenario_data['ecMaxContribution'] * adjustment_factor
    
    # Prepare forecast
    forecast_data = []
    for country in scenario_data['Country Name'].unique():
        country_df = scenario_data[scenario_data['Country Name'] == country]
        ts_data = country_df.groupby('start_year')['adjusted_funding'].sum().reset_index()
        ts_data = ts_data.rename(columns={"start_year": "ds", "adjusted_funding": "y"})
        ts_data['ds'] = pd.to_datetime(ts_data['ds'], format='%Y')

        if len(ts_data) < 3:
            continue

        m = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        m.fit(ts_data)

        future = m.make_future_dataframe(periods=years, freq='Y')
        forecast = m.predict(future)
        forecast['country'] = country
        forecast_data.append(forecast)
    
    if not forecast_data:
        return pd.DataFrame()
    
    return pd.concat(forecast_data)

def project_count_forecast(data, years):
    count_data = data.groupby('start_year').size().reset_index(name='count')
    count_data = count_data.rename(columns={"start_year": "ds", "count": "y"})
    count_data['ds'] = pd.to_datetime(count_data['ds'], format='%Y')

    if len(count_data) < 3:
        return px.line(title="Not enough data for forecasting")

    m = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True
    )
    m.fit(count_data)

    future = m.make_future_dataframe(periods=years, freq='Y')
    forecast = m.predict(future)

    fig = plot_plotly(m, forecast)
    fig.update_layout(
        title="Project Count Forecast",
        xaxis_title="Year",
        yaxis_title="Number of Projects"
    )
    return fig

# ====================
# ENHANCED COMPONENTS
# ====================
def enhanced_metrics(selected_data):
    cols = st.columns(4)
    metrics = [
        ("Total Funding", format_currency(selected_data['ecMaxContribution'].sum())),
        ("Avg. Project Size", format_currency(selected_data['ecMaxContribution'].mean())),
        ("Countries", len(selected_data['Country Name'].unique())),
        ("Projects", len(selected_data))
    ]
    
    for col, (label, value) in zip(cols, metrics):
        with col:
            with stylable_container(
                key=f"metric_{label}",
                css_styles="""
                {
                    background-color: white;
                    border-radius: 10px;
                    padding: 15px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    text-align: center;
                    transition: transform 0.3s;
                }
                :hover {
                    transform: translateY(-5px);
                    box-shadow: 0 6px 8px rgba(0,0,0,0.15);
                }
                """
            ):
                st.metric(label, value)

def predictive_analytics_tab(filtered_proj):
    st.title("🔮 Advanced Predictive Analytics")
    
    with st.expander("⚙️ Forecast Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            countries = st.multiselect(
                "Select Countries",
                filtered_proj['Country Name'].unique(),
                default=["Germany", "France", "Netherlands"]
            )
            
            topics = st.multiselect(
                "Filter Topics",
                filtered_proj['topic_label'].unique()
            )
            
        with col2:
            forecast_years = st.slider(
                "Forecast Horizon (years)",
                1, 10, 5
            )
            
            models = st.selectbox(
                "Forecast Model",
                ["Prophet (Default)", "Linear Trend", "Logistic Growth"]
            )
    
    if not countries:
        st.warning("Please select at least one country")
        return
    
    # Apply filters
    forecast_data = filtered_proj[filtered_proj['Country Name'].isin(countries)]
    if topics:
        forecast_data = forecast_data[forecast_data['topic_label'].isin(topics)]
    
    # Generate forecasts
    tab1, tab2 = st.tabs(["Funding Forecast", "Project Count Forecast"])
    
    with tab1:
        st.subheader("Funding Trend Forecast")
        fig = funding_forecast(forecast_data, forecast_years)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for forecasting")
    
    with tab2:
        st.subheader("Project Count Forecast")
        fig = project_count_forecast(forecast_data, forecast_years)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scenario modeling
    with st.expander("🧪 Scenario Modeling", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            scenarios = {
                "Budget Cut (-20%)": 0.8,
                "Baseline": 1.0,
                "Budget Increase (+20%)": 1.2,
                "Custom": None
            }
            selected_scenario = st.selectbox("Predefined Scenarios", list(scenarios.keys()))
            
            if selected_scenario == "Custom":
                custom_adjustment = st.slider("Custom Adjustment (%)", -50, 200, 0)
                adjustment_factor = 1 + custom_adjustment/100
            else:
                adjustment_factor = scenarios[selected_scenario]
        
        with col2:
            if selected_scenario != "Custom":
                st.metric(
                    "Scenario Impact",
                    f"{int((adjustment_factor-1)*100)}% change",
                    "Compared to baseline"
                )
            
            if st.button("Run Scenario Analysis"):
                scenario_results = run_scenario(forecast_data, adjustment_factor, forecast_years)
                if not scenario_results.empty:
                    st.dataframe(scenario_results[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'country']].rename(
                        columns={
                            'ds': 'Date',
                            'yhat': 'Predicted Funding',
                            'yhat_lower': 'Min Estimate',
                            'yhat_upper': 'Max Estimate'
                        }
                    ))
                else:
                    st.warning("Not enough data for scenario analysis")

def funding_forecast(data, years):
    forecast_data = []
    for country in data['Country Name'].unique():
        country_df = data[data['Country Name'] == country]
        ts_data = country_df.groupby('start_year')['ecMaxContribution'].sum().reset_index()
        ts_data = ts_data.rename(columns={"start_year": "ds", "ecMaxContribution": "y"})
        ts_data['ds'] = pd.to_datetime(ts_data['ds'], format='%Y')
        
        if len(ts_data) < 3:
            continue
        
        m = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        m.fit(ts_data)
        
        future = m.make_future_dataframe(periods=years, freq='Y')
        forecast = m.predict(future)
        forecast['country'] = country
        
        forecast_data.append(forecast)
    
    if not forecast_data:
        return None
    
    forecast_df = pd.concat(forecast_data)
    
    fig = px.line(
        forecast_df,
        x='ds',
        y='yhat',
        color='country',
        title="Funding Forecast by Country",
        labels={'yhat': 'Funding (€)', 'ds': 'Year'}
    )
    
    for country in forecast_df['country'].unique():
        country_forecast = forecast_df[forecast_df['country'] == country]
        fig.add_trace(go.Scatter(
            x=country_forecast['ds'],
            y=country_forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name=f"{country} Upper"
        ))
        fig.add_trace(go.Scatter(
            x=country_forecast['ds'],
            y=country_forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name=f"{country} Lower"
        ))
    
    return fig

def interactive_network_tab(filtered_org):
    st.title("🌐 Interactive Collaboration Network")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        min_projects = st.slider(
            "Minimum Shared Projects",
            1, 20, 3,
            help="Filter organizations with at least this many collaborations"
        )
        
        layout_algorithm = st.radio(
            "Layout Algorithm",
            ["Force Atlas", "Circular", "Kamada-Kawai"]
        )
        
        color_by = st.selectbox(
            "Color Nodes By",
            ["Country", "Organization Type", "Betweenness Centrality"]
        )
    
    with col2:
        G = create_network_graph(filtered_org, min_projects)
        
        if len(G.nodes()) > 100:
            st.warning(f"Showing {len(G.nodes())} nodes. Consider increasing the minimum projects filter.")
            G = G.subgraph(list(G.nodes())[:100])
        
        # Create PyVis network with proper iframe handling
        net = Network(
            height="700px",
            width="100%",
            notebook=True,
            cdn_resources="remote",
            bgcolor="white",
            font_color="black"
        )
        
        # Add nodes and edges
        for node in G.nodes():
            org_data = org_df[org_df['organisationID'] == node].iloc[0]
            net.add_node(
                node,
                label=org_data['shortName'],
                title=f"{org_data['name']}<br>Country: {org_data['Country']}",
                group=org_data['Country']
            )
        
        for edge in G.edges(data=True):
            net.add_edge(edge[0], edge[1], value=edge[2]['weight'])
        
        # Generate HTML directly without tempfile
        html = net.generate_html()
        html = html.replace("'", '"')  # Fix quote escaping
        
        # Render with proper iframe settings
        st.components.v1.html(
            html,
            height=700,
            scrolling=True,
            width=None,
            key=f"network_{min_projects}"
        )
        # Clean up memory
        gc.collect()

def topic_analysis_tab(filtered_proj):
    st.title("🧠 Topic Evolution Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Funding Trends", "Topic Network", "Word Clouds"])
    
    with tab1:
        st.subheader("Topic Funding Over Time")
        topic_trend = filtered_proj.groupby(['start_year', 'topic_label'])['ecMaxContribution'].sum().reset_index()
        
        fig = px.line(
            topic_trend,
            x='start_year',
            y='ecMaxContribution',
            color='topic_label',
            title="Funding Trends by Research Topic",
            labels={'ecMaxContribution': 'Total Funding (€)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Topic Collaboration Network")
        topic_org = filtered_proj.groupby(['topic_label', 'id'])['ecMaxContribution'].sum().reset_index()
        G = nx.Graph()
        
        for _, row in topic_org.iterrows():
            G.add_edge(row['topic_label'], row['id'], weight=row['ecMaxContribution'])
        
        # Visualization
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [node if node in filtered_proj['topic_label'].unique() else "" for node in G.nodes()]
        
        fig = go.Figure(
            data=[
                go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888')),
                go.Scatter(
                    x=node_x, y=node_y, mode='markers', text=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='Viridis',
                        size=10,
                        color=[G.degree(node) for node in G.nodes()],
                        line_width=2
                    )
                )
            ],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Topic Word Clouds")
        selected_topic = st.selectbox("Select Topic", filtered_proj['topic_label'].unique())
        
        topic_text = " ".join(
            filtered_proj[filtered_proj['topic_label'] == selected_topic]['cleaned_text'].dropna()
        )
        
        if topic_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(topic_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("No text data available for this topic")

# ====================
# MAIN APP
# ====================
def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("🌍 Horizon Europe PRO")
        st.markdown("""
        **Advanced analytics dashboard** for Horizon Europe funding data with predictive insights and collaboration networks.
        """)
        
        pages = {
            "Dashboard": "🏠 Overview",
            "Network Analysis": "🌐 Collaborations",
            "Predictive Analytics": "🔮 Forecasts",
            "Topic Analysis": "🧠 Research Trends",
            "Data Explorer": "📊 Raw Data"
        }
        
        selection = st.radio("Navigation", list(pages.values()))
        
        st.markdown("---")
        st.header("Global Filters")
        years = st.slider(
            "Project Years",
            int(proj_df['start_year'].min()),
            int(proj_df['start_year'].max()),
            (2021, 2027)
        )
        
        funding_cats = st.multiselect(
            "Funding Categories",
            proj_df['fundingCategory'].unique(),
            default=["ERC", "MSCA"]
        )
        
        countries = st.multiselect(
            "Countries",
            proj_df['Country Name'].unique()
        )
    
    # Apply filters
    filtered_proj = proj_df[
        (proj_df['start_year'] >= years[0]) & 
        (proj_df['start_year'] <= years[1]) &
        (proj_df['fundingCategory'].isin(funding_cats))
    ]
    
    if countries:
        filtered_proj = filtered_proj[filtered_proj['Country Name'].isin(countries)]
    
    filtered_org = org_df[org_df['projectID'].isin(filtered_proj['id'])]
    
    # Page routing
    if selection == "🏠 Overview":
        st.title("Horizon Europe Dashboard PRO")
        enhanced_metrics(filtered_proj)
        
        col1, col2 = st.columns(2)
        with col1:
            # Filter out rows with missing values for treemap
            filtered_treemap_data = filtered_proj.dropna(subset=['Country Name', 'topic_label'])
            st.plotly_chart(px.treemap(
                filtered_treemap_data,
                path=['Country Name', 'topic_label'],
                values='ecMaxContribution',
                title="Funding Distribution",
                color='ecMaxContribution',
                color_continuous_scale='Blues'
            ), use_container_width=True)
        
        with col2:
            st.plotly_chart(px.bar(
                filtered_proj.groupby('topic_label')['ecMaxContribution'].sum().reset_index().sort_values('ecMaxContribution', ascending=False).head(10),
                x='topic_label',
                y='ecMaxContribution',
                title="Top 10 Funded Research Areas",
                labels={'ecMaxContribution': 'Total Funding (€)'}
            ), use_container_width=True)
    
    elif selection == "🌐 Collaborations":
        interactive_network_tab(filtered_org)
    
    elif selection == "🔮 Forecasts":
        predictive_analytics_tab(filtered_proj)
    
    elif selection == "🧠 Research Trends":
        topic_analysis_tab(filtered_proj)
    
    elif selection == "📊 Raw Data":
        st.title("Project Data Explorer")
        
        # Using Streamlit's native data display instead of AgGrid
        cols_to_show = [
            'title', 'acronym', 'Country Name', 'topic_label',
            'startDate', 'endDate', 'ecMaxContribution', 'fundingCategory'
        ]
        
        # Add search and filter functionality
        search_col1, search_col2 = st.columns(2)
        with search_col1:
            search_term = st.text_input("Search projects")
        with search_col2:
            items_per_page = st.selectbox("Items per page", [10, 25, 50, 100])
        
        # Apply search
        if search_term:
            filtered_display = filtered_proj[
                filtered_proj['title'].str.contains(search_term, case=False) |
                filtered_proj['acronym'].str.contains(search_term, case=False) |
                filtered_proj['topic_label'].str.contains(search_term, case=False)
            ][cols_to_show]
        else:
            filtered_display = filtered_proj[cols_to_show]
        
        # Pagination
        page_number = st.number_input("Page number", min_value=1, 
                                    max_value=len(filtered_display)//items_per_page + 1, 
                                    value=1)
        start_idx = (page_number - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        # Display the data
        st.dataframe(
            filtered_display.iloc[start_idx:end_idx],
            column_config={
                "ecMaxContribution": st.column_config.NumberColumn(
                    "Funding (€)",
                    format="%.2f"
                ),
                "startDate": st.column_config.DateColumn("Start Date"),
                "endDate": st.column_config.DateColumn("End Date")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        st.download_button(
            label="Download filtered data as CSV",
            data=filtered_display.to_csv(index=False).encode('utf-8'),
            file_name="horizon_projects_filtered.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
