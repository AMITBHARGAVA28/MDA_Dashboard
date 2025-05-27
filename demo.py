# -*- coding: utf-8 -*-
"""Horizon Europe Meta Dashboard - Combined Version"""

# ====================
# IMPORTS & CONFIG
# ====================
import streamlit as st
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
from collections import defaultdict
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    layout="wide",
    page_title="Horizon Europe Meta Dashboard",
    page_icon="🌐",
    initial_sidebar_state="expanded"
)

# ====================
# DATA LOADING
# ====================
@st.cache_data(ttl=3600)
def load_data():
    # Load with encoding fallback
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    for encoding in encodings:
        try:
            org_df = pd.read_csv("horizon_organizations.csv", encoding=encoding)
            proj_df = pd.read_csv("horizon_projects.csv", encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    # Date processing
    proj_df['startDate'] = pd.to_datetime(proj_df['startDate'], errors='coerce')
    proj_df['endDate'] = pd.to_datetime(proj_df['endDate'], errors='coerce')
    proj_df['start_year'] = proj_df['startDate'].dt.year
    
    # Enhanced funding categories (from PRO)
    funding_scheme_groups = {
        "ERC": ["ERC", "HORIZON-ERC"],
        "MSCA": ["MSCA", "HORIZON-TMA-MSCA"],
        "EIC": ["EIC", "HORIZON-EIC"],
        "RIA/IA": ["RIA", "IA", "HORIZON-RIA"],
        "CSA": ["CSA", "HORIZON-CSA"],
        "Other": []
    }
    
    proj_df['fundingCategory'] = "Other"
    for category, schemes in funding_scheme_groups.items():
        for scheme in schemes:
            proj_df.loc[proj_df['fundingScheme'].str.contains(scheme, na=False), 'fundingCategory'] = category
    
    # Topic clustering (cached separately)
    @st.cache_data
    def cluster_topics(texts):
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts.fillna(''))
        return KMeans(n_clusters=8).fit_predict(X)
    
    proj_df['topic_cluster'] = cluster_topics(proj_df['cleaned_text'])
    
    return org_df, proj_df

org_df, proj_df = load_data()

# ====================
# UTILITY FUNCTIONS
# ====================
def format_currency(value):
    """Improved currency formatting from PRO version"""
    if pd.isna(value):
        return "€0"
    if value >= 1e9:
        return f"€{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"€{value/1e6:.1f}M"
    else:
        return f"€{value/1e3:,.0f}K"

@st.cache_data
def build_collaboration_network(_org_df, min_projects=3):
    """Combined network builder with PRO's filtering"""
    project_orgs = _org_df.groupby('projectID')['organisationID'].apply(set)
    edge_counter = defaultdict(int)
    
    for orgs in project_orgs:
        for u, v in combinations(sorted(orgs), 2):
            edge_counter[(u, v)] += 1
    
    G = nx.Graph()
    G.add_edges_from((u, v, {'weight': w}) for (u, v), w in edge_counter.items())
    
    # Filter nodes by minimum projects
    degrees = dict(G.degree())
    filtered_nodes = [n for n, d in degrees.items() if d >= min_projects]
    return G.subgraph(filtered_nodes)

def create_network_visualization(G, color_by="Country"):
    """Enhanced version with PRO's grouping options"""
    pos = nx.spring_layout(G, k=0.15, iterations=30, seed=42)
    
    # Edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(150,150,150,0.5)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node traces with grouping
    node_data = []
    for node in G.nodes():
        meta = org_df[org_df['organisationID'] == node].iloc[0].to_dict()
        
        if color_by == "Country":
            color_val = meta.get('Country', 'Unknown')
        elif color_by == "Organization Type":
            color_val = meta.get('role', 'Unknown')
        else:  # Betweenness
            color_val = nx.betweenness_centrality(G).get(node, 0)
        
        node_data.append((
            pos[node][0], pos[node][1],
            f"<b>{meta.get('name',str(node))}</b><br>"
            f"Country: {meta.get('Country','Unknown')}<br>"
            f"Projects: {G.degree(node)}",
            color_val,
            5 + (G.degree(node) * 0.5)
        ))
    
    node_x, node_y, text, colors, sizes = zip(*node_data)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=colors,
            size=sizes,
            line_width=0.5
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            height=700
        )
    )
    return fig

# ====================
# PREDICTIVE FUNCTIONS
# ====================
def generate_forecast(data, years=5):
    """From PRO version with stability improvements"""
    if len(data) < 3:
        return None
    
    ts_data = data.groupby('start_year')['ecMaxContribution'].sum().reset_index()
    ts_data.columns = ['ds', 'y']
    ts_data['ds'] = pd.to_datetime(ts_data['ds'], format='%Y')
    
    model = Prophet(seasonality_mode='multiplicative')
    model.fit(ts_data)
    
    future = model.make_future_dataframe(periods=years, freq='Y')
    forecast = model.predict(future)
    
    fig = plot_plotly(model, forecast)
    fig.update_layout(
        title=f"Funding Forecast ({years} years)",
        xaxis_title="Year",
        yaxis_title="Funding (€)"
    )
    return fig

# ====================
# DASHBOARD SECTIONS
# ====================
def show_overview(filtered_proj):
    """Combined overview with enhanced metrics"""
    st.title("🌍 Horizon Europe Dashboard")
    
    # Enhanced metrics from PRO
    cols = st.columns(4)
    metrics = [
        ("Total Funding", filtered_proj['ecMaxContribution'].sum()),
        ("Avg. Funding", filtered_proj['ecMaxContribution'].mean()),
        ("Projects", len(filtered_proj)),
        ("Countries", filtered_proj['Country Name'].nunique())
    ]
    
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, 
                     format_currency(value) if label != "Projects" else f"{value:,}",
                     help=f"Filtered {label.lower()}")

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.treemap(
            filtered_proj,
            path=['Country Name', 'topic_label'],
            values='ecMaxContribution',
            title="Funding Distribution",
            color='ecMaxContribution',
            color_continuous_scale='Blues'
        ), use_container_width=True)
    
    with col2:
        st.plotly_chart(px.bar(
            filtered_proj.groupby('topic_label')['ecMaxContribution'].sum()
                           .reset_index()
                           .sort_values('ecMaxContribution', ascending=False)
                           .head(10),
            x='topic_label',
            y='ecMaxContribution',
            title="Top 10 Funded Topics",
            labels={'ecMaxContribution': 'Total Funding'}
        ), use_container_width=True)

def show_network(filtered_org):
    """Enhanced network visualization"""
    st.title("🔗 Collaboration Network")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        min_projects = st.slider(
            "Minimum shared projects",
            1, 20, 3,
            help="Filter organizations with at least this many collaborations"
        )
        color_by = st.selectbox(
            "Color nodes by",
            ["Country", "Organization Type", "Project Count"]
        )
    
    with col2:
        G = build_collaboration_network(filtered_org, min_projects)
        if len(G.nodes) == 0:
            st.warning("No organizations meet the current filters")
        else:
            st.plotly_chart(
                create_network_visualization(G, color_by),
                use_container_width=True
            )

def show_predictive(filtered_proj):
    """From PRO version with simplified interface"""
    st.title("🔮 Predictive Analytics")
    
    years = st.slider(
        "Forecast horizon (years)",
        1, 10, 5
    )
    
    tab1, tab2 = st.tabs(["Funding Forecast", "Topic Trends"])
    with tab1:
        if len(filtered_proj) < 10:
            st.warning("Not enough data for forecasting")
        else:
            st.plotly_chart(
                generate_forecast(filtered_proj, years),
                use_container_width=True
            )
    
    with tab2:
        selected_topic = st.selectbox(
            "Select topic to analyze",
            filtered_proj['topic_label'].unique()
        )
        topic_data = filtered_proj[filtered_proj['topic_label'] == selected_topic]
        
        st.plotly_chart(px.line(
            topic_data.groupby('start_year')['ecMaxContribution'].sum().reset_index(),
            x='start_year',
            y='ecMaxContribution',
            title=f"Funding Trend for {selected_topic}"
        ), use_container_width=True)

def show_explorer(filtered_proj):
    """Enhanced data explorer from PRO"""
    st.title("📊 Project Explorer")
    
    # Search functionality
    search_term = st.text_input("Search projects by title or topic")
    if search_term:
        filtered_proj = filtered_proj[
            filtered_proj['title'].str.contains(search_term, case=False) |
            filtered_proj['topic_label'].str.contains(search_term, case=False)
        ]
    
    # Display with improved formatting
    st.dataframe(
        filtered_proj[[
            'title', 'acronym', 'Country Name', 
            'topic_label', 'startDate', 'endDate',
            'ecMaxContribution', 'fundingCategory'
        ]],
        column_config={
            "ecMaxContribution": st.column_config.NumberColumn(
                "Funding",
                format="€%.2f"
            ),
            "startDate": "Start Date",
            "endDate": "End Date"
        },
        use_container_width=True,
        height=500
    )
    
    # Download button
    st.download_button(
        "Export filtered data",
        filtered_proj.to_csv(index=False).encode('utf-8'),
        "horizon_filtered.csv"
    )

# ====================
# MAIN APP
# ====================
def main():
    # Sidebar filters
    with st.sidebar:
        st.title("Filters")
        
        years = st.slider(
            "Project years",
            int(proj_df['start_year'].min()),
            int(proj_df['start_year'].max()),
            (2021, 2027)
        )
        
        countries = st.multiselect(
            "Countries",
            proj_df['Country Name'].unique()
        )
        
        funding_cats = st.multiselect(
            "Funding categories",
            proj_df['fundingCategory'].unique()
        )
        
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Overview", "Network Analysis", "Predictive Analytics", "Data Explorer"]
        )
    
    # Apply filters
    filtered_proj = proj_df[
        (proj_df['start_year'] >= years[0]) & 
        (proj_df['start_year'] <= years[1])
    ]
    
    if countries:
        filtered_proj = filtered_proj[filtered_proj['Country Name'].isin(countries)]
    if funding_cats:
        filtered_proj = filtered_proj[filtered_proj['fundingCategory'].isin(funding_cats)]
    
    filtered_org = org_df[org_df['projectID'].isin(filtered_proj['id'])]
    
    # Page routing
    if page == "Overview":
        show_overview(filtered_proj)
    elif page == "Network Analysis":
        show_network(filtered_org)
    elif page == "Predictive Analytics":
        show_predictive(filtered_proj)
    elif page == "Data Explorer":
        show_explorer(filtered_proj)

if __name__ == "__main__":
    main()
