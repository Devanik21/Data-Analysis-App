import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import duckdb
import requests
from bs4 import BeautifulSoup
import io
import base64
from datetime import datetime
import warnings
import sys
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import openpyxl
from sqlalchemy import create_engine, text
import time
import re
import json
import pickle
from typing import Optional, Dict, Any, List
from collections import Counter
import google.generativeai as genai
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Advanced Data Analysis Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* General App Styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f4f6f8; /* Light background for the main area */
    }
    .main-header {
        font-size: 3rem;
        color: #1a5276; /* Darker, more professional blue */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px #cccccc;
    }
    .tool-header {
        font-size: 1.5rem;
        color: #c0392b; /* Professional red for emphasis */
        margin: 1rem 0;
        border-bottom: 3px solid #c0392b;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    /* Specific adjustments for dark background */
    .css-18e3th9 { /* Main content area */
        background-color: #1e1e1e; /* Dark background */
        color: #e0e0e0; /* Light text */
    }
    .metric-card {
        background-color: #ffffff; /* White cards for metrics */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1a5276; /* Accent border */
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stButton>button {
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'sql_history' not in st.session_state:
    st.session_state.sql_history = []
if 'python_history' not in st.session_state:
    st.session_state.python_history = []
if 'current_query' not in st.session_state: # For SQL Query tool
    st.session_state.current_query = 'SELECT * FROM data LIMIT 10;'
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'python_plots' not in st.session_state:
    st.session_state.python_plots = []
if 'saved_sql_queries' not in st.session_state:
    st.session_state.saved_sql_queries = {}
if 'python_plotly_figs' not in st.session_state:
    st.session_state.python_plotly_figs = []

# Main Title
st.markdown('<h1 class="main-header">🔬 Advanced Data Analysis Suite</h1>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input("Google AI API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        st.sidebar.success("Gemini model configured!")
    except Exception as e:
        st.sidebar.error(f"Invalid API Key: {e}")

# --- Session Management ---
st.sidebar.title("💾 Session Management")

# Create a dictionary of the current session state to save
# We exclude non-serializable or re-initializable objects like the gemini model
# Also exclude temporary results that can be regenerated
state_to_save = {
    key: value for key, value in st.session_state.items() 
    if key not in ['gemini_model', 'python_plots', 'python_plotly_figs', 'sql_result']
}

# Serialize the state
try:
    # Use a higher protocol for efficiency with large dataframes
    state_bytes = pickle.dumps(state_to_save, protocol=pickle.HIGHEST_PROTOCOL)
    st.sidebar.download_button(
        label="📥 Save Session",
        data=state_bytes,
        file_name=f"data_suite_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
        mime="application/octet-stream",
        help="Save your current data, history, and code to a file."
    )
except Exception as e:
    st.sidebar.error(f"Error preparing session for download: {e}")

# Load session state
loaded_session_file = st.sidebar.file_uploader(
    "📤 Load Session", 
    type=['pkl'], 
    help="Load a previously saved session file (.pkl)."
)

if loaded_session_file is not None:
    try:
        loaded_state = pickle.load(loaded_session_file)
        for key, value in loaded_state.items():
            st.session_state[key] = value
        st.sidebar.success("Session loaded successfully! Refreshing app...")
        st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"Error loading session file: {e}. The file may be corrupted or from an incompatible version.")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
tools = [
    "📤 Data Upload",
    "🔍 SQL Query Engine",
    "📊 Exploratory Data Analysis (EDA)",
    "📈 Excel Query Tool",
    "💼 Power BI Style Dashboard",
    "🐼 Pandas Query Tool",
    "🐍 Python Advanced Analytics",
    "🌐 Web Scraping Tool",
    "🤖 AI-Powered Insights (Gemini)"
]

selected_tool = st.sidebar.selectbox("Select Analysis Tool", tools)

# Helper Functions
@st.cache_data
def load_data(file: Any) -> Optional[pd.DataFrame]:
    """Load data from various file formats"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        elif file.name.endswith('.parquet'):
            df = pd.read_parquet(file)
        else:
            st.error("Unsupported file format")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_download_link(df: pd.DataFrame, filename: str = "data.csv") -> str:
    """Create download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def execute_sql_query(df: pd.DataFrame, query: str) -> tuple[Optional[pd.DataFrame], Optional[str], float]:
    """Execute SQL query on dataframe using DuckDB for high performance."""
    start_time = time.time()
    conn = duckdb.connect(database=':memory:')
    conn.register('data', df)
    
    query_upper = query.strip().upper()
    is_dml = any(query_upper.startswith(s) for s in ['INSERT', 'UPDATE', 'DELETE'])

    try:
        cursor = conn.execute(query)
        duration = time.time() - start_time
        
        if is_dml:
            # For DML, the "result" is the entire updated table.
            result_df = conn.execute('SELECT * FROM data').fetchdf()
        else:
            # For SELECT, EXPLAIN, etc., fetch the result from the original cursor
            result_df = cursor.fetchdf()
            
        conn.close()
        return result_df, None, duration
    except Exception as e:
        conn.close()
        duration = time.time() - start_time
        return None, str(e), duration

def advanced_outlier_detection(df: pd.DataFrame, column: str) -> dict:
    """Advanced outlier detection using multiple methods"""
    methods = {}
    
    # IQR Method
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    methods['IQR'] = len(iqr_outliers)
    
    # Z-Score Method
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    z_outliers = df[z_scores > 3]
    methods['Z-Score'] = len(z_outliers)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outlier_labels = iso_forest.fit_predict(df[[column]].dropna())
    iso_outliers = sum(outlier_labels == -1)
    methods['Isolation Forest'] = iso_outliers    
    
    # Local Outlier Factor (LOF)
    # LOF needs more than 1 sample
    if len(df[[column]].dropna()) > 5:
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=min(20, len(df[[column]].dropna())-1), contamination='auto')
        lof_labels = lof.fit_predict(df[[column]].dropna())
        lof_outliers = sum(lof_labels == -1)
        methods['Local Outlier Factor'] = lof_outliers

    return methods

def generate_chart(df: pd.DataFrame, config: dict, title: str) -> None:
    """Generate chart based on configuration"""
    try:
        if config['type'] == 'Bar':
            # Ensure x and y are serializable
            x = df[config['x']].astype(str) if pd.api.types.is_object_dtype(df[config['x']]) else df[config['x']]
            y = df[config['y']].astype(float) if not pd.api.types.is_numeric_dtype(df[config['y']]) else df[config['y']]
            color = config.get('color')
            if color:
                color_data = df[color].astype(str) if pd.api.types.is_object_dtype(df[color]) else df[color]
            else:
                color_data = None
            fig = px.bar(df, x=x, y=y, color=color_data, title=title)
        elif config['type'] == 'Line':
            x = df[config['x']].astype(str) if pd.api.types.is_object_dtype(df[config['x']]) else df[config['x']]
            y = df[config['y']].astype(float) if not pd.api.types.is_numeric_dtype(df[config['y']]) else df[config['y']]
            color = config.get('color')
            if color:
                color_data = df[color].astype(str) if pd.api.types.is_object_dtype(df[color]) else df[color]
            else:
                color_data = None
            fig = px.line(df, x=x, y=y, color=color_data, title=title)
        elif config['type'] == 'Scatter':
            x = df[config['x']].astype(float) if not pd.api.types.is_numeric_dtype(df[config['x']]) else df[config['x']]
            y = df[config['y']].astype(float) if not pd.api.types.is_numeric_dtype(df[config['y']]) else df[config['y']]
            color = config.get('color')
            if color:
                color_data = df[color].astype(str) if pd.api.types.is_object_dtype(df[color]) or not pd.api.types.is_numeric_dtype(df[color]) else df[color]
            else:
                color_data = None
            fig = px.scatter(df, x=x, y=y, color=color_data, title=title)
        elif config['type'] == 'Histogram':
            col = df[config['column']]
            if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
                col = col.astype(float)
            else:
                col = col.astype(str)
            fig = px.histogram(df, x=col, title=title)
        elif config['type'] == 'Box':
            col = df[config['column']]
            if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
                col = col.astype(float)
            else:
                col = col.astype(str)
            fig = px.box(df, y=col, title=title)
        elif config['type'] == 'Pie':
            value_counts = df[config['column']].value_counts().head(10)
            fig = px.pie(values=value_counts.values.astype(float), names=value_counts.index.astype(str), title=title)
        elif config['type'] == 'Heatmap':
            if config['columns']:
                corr_matrix = df[config['columns']].corr()
                fig = px.imshow(corr_matrix.astype(float), title=title)
            else:
                st.warning("No numeric columns available for heatmap")
                return
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating chart: {str(e)}")

def generate_gemini_content(prompt_text: str) -> Optional[str]:
    """Generates content using the Gemini model."""
    if not st.session_state.gemini_model:
        st.error("Gemini model not configured. Please enter your API key in the sidebar.")
        return None
    try:
        with st.spinner("🤖 Gemini is thinking..."):
            response = st.session_state.gemini_model.generate_content(prompt_text)
            return response.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return None

def execute_python_code(code: str, df: pd.DataFrame) -> None:
    """
    Execute Python code safely and capture output, matplotlib plots, and plotly figures.
    The user's code can create a variable named 'fig' (for a single plot) or
    'figs' (for a list of plots) to have them displayed.
    """
    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = output_buffer

    # Clear previous plots from session state
    st.session_state.python_plots = []
    st.session_state.python_plotly_figs = []

    try:
        # Create a safe execution environment
        exec_globals = {
            'df': df.copy(), # Use a copy to prevent modification of the original df in session state
            'pd': pd,
            'np': np,
            'plt': plt,
            'px': px,
            'go': go,
            'sns': sns, # Add seaborn
            'stats': stats,
            'StandardScaler': StandardScaler,
            'PCA': PCA,
            'KMeans': KMeans,
            'IsolationForest': IsolationForest,
            'LabelEncoder': LabelEncoder,
            'st': st # Allow users to use some streamlit functions if they want
        }

        # Execute code
        exec(code, exec_globals)

        # Capture any matplotlib/seaborn plots created
        fig_nums = plt.get_fignums()
        for i in fig_nums:
            fig = plt.figure(i)
            st.session_state.python_plots.append(fig)

        # Capture any plotly figure object(s) created by the user
        if 'fig' in exec_globals:
            if isinstance(exec_globals['fig'], go.Figure):
                st.session_state.python_plotly_figs.append(exec_globals['fig'])
        if 'figs' in exec_globals:
            if isinstance(exec_globals['figs'], list):
                for f in exec_globals['figs']:
                    if isinstance(f, go.Figure):
                        st.session_state.python_plotly_figs.append(f)

    except Exception as e:
        st.session_state.python_output = f"Execution Error: {str(e)}"
    else:
        st.session_state.python_output = output_buffer.getvalue()
    finally:
        # Restore stdout
        sys.stdout = original_stdout

def scrape_website(url: str, method: str, params: dict, export_format: str) -> None:
    """Perform web scraping based on selected method"""
    try:
        headers_dict = {
            'User-Agent': params.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        } if params.get('headers') else {}
        
        with st.spinner("Scraping website..."):
            # Make request
            time.sleep(float(params.get('delay', 1.0)))
            response = requests.get(url, headers=headers_dict, timeout=params.get('timeout', 10))
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            exec_globals = {'soup': soup, 'results': [], 'pd': pd, 'np': np, 're': re} # Define exec_globals for custom BS
            
            if method == "Extract Text by Tag":
                tag = params.get('tag', 'p')
                class_name = params.get('class_name', '')
                limit = params.get('limit', 10)
                
                if class_name:
                    elements = soup.find_all(tag, class_=class_name)
                else:
                    elements = soup.find_all(tag)
                
                for elem in elements[:limit]:
                    results.append({
                        'text': elem.get_text().strip(),
                        'tag': tag,
                        'html': str(elem)
                    })
            
            elif method == "Extract by CSS Selector":
                selector = params.get('selector', 'p')
                limit = params.get('limit', 10)
                
                elements = soup.select(selector)
                for elem in elements[:limit]:
                    results.append({
                        'text': elem.get_text().strip(),
                        'selector': selector,
                        'html': str(elem)
                    })
            
            elif method == "Extract Table Data":
                table_index = params.get('table_index', 0)
                tables = soup.find_all('table')
                
                if tables and len(tables) > table_index:
                    table = tables[table_index]
                    rows = table.find_all('tr')
                    
                    for i, row in enumerate(rows):
                        cells = row.find_all(['td', 'th'])
                        row_data = [cell.get_text().strip() for cell in cells]
                        results.append({
                            'row_index': i,
                            'data': row_data,
                            'html': str(row)
                        })
                else:
                    st.error(f"Table {table_index} not found!")
                    return
            
            elif method == "Extract All Links":
                filter_text = params.get('filter_text', '')
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link['href']
                    text = link.get_text().strip()
                    
                    if not filter_text or filter_text.lower() in text.lower():
                        results.append({
                            'url': href,
                            'text': text,
                            'full_url': requests.compat.urljoin(url, href)
                        })
            
            elif method == "Extract Images":
                min_width = params.get('min_width', 0)
                images = soup.find_all('img')
                
                for img in images:
                    src = img.get('src', '')
                    alt = img.get('alt', '')
                    width = img.get('width', 0)
                    
                    try:
                        width = int(width) if width else 0
                    except:
                        width = 0
                    
                    if width >= min_width:
                        results.append({
                            'src': src,
                            'alt': alt,
                            'width': width,
                            'full_url': requests.compat.urljoin(url, src)
                        })
            
            elif method == "Custom BeautifulSoup":
                # Execute custom code
                custom_code = params.get('custom_code', '')
                # Execute the user's code
                # The user's code is expected to populate the 'results' list or create a 'df_results' DataFrame
                exec(custom_code, exec_globals)
                results = exec_globals.get('results', [])
            
            # Display results
            if results:
                st.success(f"Successfully scraped {len(results)} items!")
                
                # Convert to DataFrame for better display
                # Check if the custom code created a DataFrame named 'df_results'
                if 'df_results' in exec_globals and isinstance(exec_globals['df_results'], pd.DataFrame):
                    df_results = exec_globals['df_results']
                    st.subheader("Custom Code Results (DataFrame)")
                    st.dataframe(df_results)
                elif results and isinstance(results[0], dict):
                    df_results = pd.DataFrame(results)
                    st.dataframe(df_results)
                    
                    # Export options
                    if export_format == "CSV":
                        csv_data = df_results.to_csv(index=False)
                        st.download_button(
                            "💾 Download CSV",
                            csv_data,
                            file_name=f"scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    elif export_format == "JSON":
                        json_data = df_results.to_json(orient='records')
                        st.download_button(
                            "💾 Download JSON",
                            data=json_data,
                            file_name=f"scraped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
    except Exception as e:
        st.error(f"Web scraping error: {str(e)}")
# Tool Implementation

if selected_tool == "📤 Data Upload": # Keep this as the first tool
    st.markdown('<h2 class="tool-header">📤 Data Upload & Preview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )
        
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.success(f"Successfully loaded {uploaded_file.name}")
                    
                    # Data Preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(10))
                    
                    # Download processed data
                    st.html(
 create_download_link(df, f"processed_{uploaded_file.name}")
                    )
    
    with col2:
        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Rows", f"{len(df):,}")
            st.metric("📈 Columns", len(df.columns))
            st.metric("💾 Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.metric("🕒 Data Types", len(df.dtypes.unique()))
            st.markdown('</div>', unsafe_allow_html=True)
            
    if st.session_state.df is not None:
        df = st.session_state.df # Ensure df is the one from session state
        st.markdown("---")
        st.subheader("🔬 Detailed Data Insights & Tools")

        with st.expander("📊 Column-wise Summary", expanded=True):
            summary_data = []
            for col in df.columns:
                col_data = {
                    "Column": col,
                    "Data Type": str(df[col].dtype),
                    "Missing Values": df[col].isnull().sum(),
                    "Missing (%)": f"{df[col].isnull().mean() * 100:.2f}%",
                    "Unique Values": df[col].nunique()
                }
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_data["Mean"] = f"{df[col].mean():.2f}"
                    col_data["Median"] = f"{df[col].median():.2f}"
                    col_data["Std Dev"] = f"{df[col].std():.2f}"
                    col_data["Min"] = f"{df[col].min():.2f}"
                    col_data["Max"] = f"{df[col].max():.2f}"
                else: # For categorical, show top few unique values
                    top_values = df[col].value_counts().nlargest(3).index.tolist()
                    col_data["Top Values"] = ", ".join(map(str,top_values)) + ('...' if df[col].nunique() > 3 else '')
                summary_data.append(col_data)
            
            summary_df = pd.DataFrame(summary_data).set_index("Column")
            st.dataframe(summary_df)

        with st.expander("🧹 Data Cleaning Utilities"):
            st.markdown("#### 💧 Handle Missing Values")
            missing_col = st.selectbox("Select column with missing values", 
                                       [col for col in df.columns if df[col].isnull().any()], 
                                       key="clean_missing_col")
            if missing_col:
                fill_method = st.selectbox("Method", ["None", "Fill with Mean (Numeric)", "Fill with Median (Numeric)", "Fill with Mode", "Drop Rows with NaN in this column"], key="clean_fill_method")
                if st.button("Apply Missing Value Treatment", key="clean_apply_missing"):
                    df_cleaned = df.copy()
                    if fill_method == "Fill with Mean (Numeric)":
                        if pd.api.types.is_numeric_dtype(df_cleaned[missing_col]):
                            df_cleaned[missing_col].fillna(df_cleaned[missing_col].mean(), inplace=True)
                            st.session_state.df = df_cleaned
                            st.success(f"Filled NaNs in '{missing_col}' with mean.")
                        else:
                            st.error("Mean imputation only for numeric columns.")
                    elif fill_method == "Fill with Median (Numeric)":
                        if pd.api.types.is_numeric_dtype(df_cleaned[missing_col]):
                            df_cleaned[missing_col].fillna(df_cleaned[missing_col].median(), inplace=True)
                            st.session_state.df = df_cleaned
                            st.success(f"Filled NaNs in '{missing_col}' with median.")
                        else:
                            st.error("Median imputation only for numeric columns.")
                    elif fill_method == "Fill with Mode":
                        df_cleaned[missing_col].fillna(df_cleaned[missing_col].mode()[0], inplace=True)
                        st.session_state.df = df_cleaned
                        st.success(f"Filled NaNs in '{missing_col}' with mode.")
                    elif fill_method == "Drop Rows with NaN in this column":
                        df_cleaned.dropna(subset=[missing_col], inplace=True)
                        st.session_state.df = df_cleaned
                        st.success(f"Dropped rows with NaNs in '{missing_col}'. New shape: {df_cleaned.shape}")
                    st.experimental_rerun()

            st.markdown("#### 🔄 Change Data Type")
            type_col = st.selectbox("Select column to change type", df.columns.tolist(), key="clean_type_col")
            if type_col:
                new_type = st.selectbox("New Data Type", ["object (string)", "int64", "float64", "datetime64"], key="clean_new_type")
                if st.button("Convert Data Type", key="clean_apply_type"):
                    try:
                        df_typed = df.copy()
                        if new_type == "datetime64":
                            df_typed[type_col] = pd.to_datetime(df_typed[type_col])
                        else:
                            df_typed[type_col] = df_typed[type_col].astype(new_type)
                        st.session_state.df = df_typed
                        st.success(f"Converted '{type_col}' to {new_type}.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error converting type: {e}")

            st.markdown("#### 🗑️ Remove Duplicates")
            if st.button("Remove All Duplicate Rows", key="clean_remove_duplicates"):
                df_no_duplicates = df.drop_duplicates()
                removed_count = len(df) - len(df_no_duplicates)
                st.session_state.df = df_no_duplicates
                st.success(f"Removed {removed_count} duplicate rows. New shape: {df_no_duplicates.shape}")
                st.experimental_rerun()

            st.markdown("#### ✏️ Rename Columns")
            col_to_rename = st.selectbox("Select column to rename", df.columns.tolist(), key="rename_col_select")
            if col_to_rename:
                new_col_name_rename = st.text_input("Enter new name for the column:", value=col_to_rename, key="rename_col_new_name")
                if st.button("Rename Column", key="rename_col_button"):
                    if new_col_name_rename and new_col_name_rename != col_to_rename:
                        df_renamed = df.copy()
                        df_renamed.rename(columns={col_to_rename: new_col_name_rename}, inplace=True)
                        st.session_state.df = df_renamed
                        st.success(f"Column '{col_to_rename}' renamed to '{new_col_name_rename}'.")
                        st.experimental_rerun()
                    else:
                        st.warning("Please enter a valid new column name different from the original.")
            
            st.markdown("#### ❌ Drop Columns")
            cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist(), key="drop_cols_multiselect")
            if st.button("Drop Selected Columns", key="drop_cols_button"):
                if cols_to_drop:
                    df_dropped = df.drop(columns=cols_to_drop)
                    st.session_state.df = df_dropped
                    st.success(f"Dropped columns: {', '.join(cols_to_drop)}. New shape: {df_dropped.shape}")
                    st.experimental_rerun()
                else:
                    st.warning("Please select at least one column to drop.")

        with st.expander("📊 Quick Visualizations"):
            st.markdown("#### Missing Values Distribution")
            missing_counts = df.isnull().sum()
            missing_counts = missing_counts[missing_counts > 0]
            if not missing_counts.empty:
                fig_missing = px.bar(missing_counts, x=missing_counts.index, y=missing_counts.values,
                                     labels={'x': 'Column', 'y': 'Number of Missing Values'},
                                     title="Missing Values per Column")
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.info("No missing values to visualize.")

            st.markdown("#### Numeric Column Histogram")
            numeric_cols_viz = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols_viz:
                hist_col = st.selectbox("Select numeric column for histogram", numeric_cols_viz, key="viz_hist_col")
                if hist_col:
                    numeric_chart_type = st.selectbox("Select Chart Type for Numeric Column", 
                                                      ["Histogram", "Box Plot", "Violin Plot"], 
                                                      key="viz_numeric_chart_type")
                    if numeric_chart_type == "Histogram":
                        fig_numeric = px.histogram(df, x=hist_col, title=f"Histogram of {hist_col}", marginal="box")
                    elif numeric_chart_type == "Box Plot":
                        fig_numeric = px.box(df, y=hist_col, title=f"Box Plot of {hist_col}")
                    elif numeric_chart_type == "Violin Plot":
                        fig_numeric = px.violin(df, y=hist_col, title=f"Violin Plot of {hist_col}", box=True, points="all")
                    st.plotly_chart(fig_numeric, use_container_width=True)
            else:
                st.info("No numeric columns for histogram.")

            st.markdown("#### Categorical Column Bar Chart")
            cat_cols_viz = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols_viz:
                bar_col = st.selectbox("Select categorical column for visualization", cat_cols_viz, key="viz_bar_col")
                if bar_col:
                    cat_chart_type = st.selectbox("Select Chart Type for Categorical Column",
                                                  ["Bar Chart", "Pie Chart"],
                                                  key="viz_cat_chart_type")
                    
                    top_n_cat = st.slider("Show Top N categories (for Bar/Pie)", 1, 20, min(10, df[bar_col].nunique()), key="viz_cat_top_n")
                    val_counts = df[bar_col].value_counts().nlargest(top_n_cat)

                    if cat_chart_type == "Bar Chart":
                        fig_cat = px.bar(val_counts, x=val_counts.index, y=val_counts.values,
                                         labels={'x': bar_col, 'y': 'Count'}, title=f"Top {top_n_cat} Value Counts for {bar_col}")
                    elif cat_chart_type == "Pie Chart":
                        fig_cat = px.pie(val_counts, names=val_counts.index, values=val_counts.values,
                                         title=f"Top {top_n_cat} Value Counts for {bar_col}")
                    st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No categorical columns for bar chart.")

elif selected_tool == "🔍 SQL Query Engine":
    st.markdown('<h2 class="tool-header">⚡ Ultimate SQL Query Engine</h2>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.df

        # Initialize session state for parameters
        if 'sql_params' not in st.session_state:
            st.session_state.sql_params = [{"Parameter": ":min_age", "Value": "30"}]
        if 'saved_sql_queries' not in st.session_state:
            st.session_state.saved_sql_queries = {}

        main_col, side_col = st.columns([3, 1])
        
        with side_col:
            st.subheader("🛠️ Tools & Info")

            # Schema Viewer
            with st.expander("📄 Table Schema (`data`)", expanded=True):
                schema_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str)
                })
                st.dataframe(schema_info, use_container_width=True)

            # Parameterized Queries
            st.subheader("⚙️ Parameterized Query")
            st.info("Define parameters (e.g., `:my_param`) and use them in your query. They will be replaced with their values before execution.")
            edited_params = st.data_editor(
                st.session_state.sql_params,
                num_rows="dynamic",
                column_config={
                    "Parameter": st.column_config.TextColumn("Parameter", help="Parameter name, e.g., :city_name", required=True),
                    "Value": st.column_config.TextColumn("Value", help="Value to substitute", required=True)
                },
                key="sql_params_editor"
            )
            st.session_state.sql_params = edited_params

            # AI Query Generator
            st.subheader("🤖 AI Query Assistant")
            if not st.session_state.gemini_model:
                st.warning("Enter your Google AI API Key in the sidebar to enable the AI Assistant.")
            else:
                with st.expander("Generate SQL from Natural Language"):
                    nl_query = st.text_area(
                        "Describe what you want to query in plain English:",
                        placeholder="e.g., 'show me the average income by city for users older than 30, sorted by income'",
                        height=100,
                        key="sql_ai_query"
                    )
                    if st.button("✨ Generate SQL with AI"):
                        if nl_query:
                            schema_str = pd.DataFrame({'Column': df.columns, 'DataType': df.dtypes.astype(str)}).to_string()
                            prompt = f"""You are an expert SQL developer.
    Given a table named `data` with the following schema:
    {schema_str}

    Write a SQL query to answer the following question:
    "{nl_query}"

    Provide only the SQL code in a single code block, without any explanation or surrounding text.
    """
                            generated_sql = generate_gemini_content(prompt)
                            if generated_sql:
                                cleaned_sql = re.sub(r"```(sql)?\n", "", generated_sql)
                                cleaned_sql = re.sub(r"```", "", cleaned_sql).strip()
                                st.session_state.current_query = cleaned_sql
                                st.success("AI-generated SQL populated in the editor!")
                                st.experimental_rerun() # Rerun to update the text_area
                        else:
                            st.warning("Please enter a description for the AI to generate a query.")

            # Query History
            st.subheader("📚 Query History")
            if st.session_state.sql_history:
                for i, hist in enumerate(reversed(st.session_state.sql_history[-5:])):
                    with st.expander(f"Query {len(st.session_state.sql_history) - i} ({hist['timestamp']})"):
                        st.code(hist['query'], language='sql')
                        st.caption(f"Rows: {hist['rows']} | Duration: {hist.get('duration', 0.0):.4f}s")
                        if st.button("Reuse this query", key=f"reuse_sql_{i}"):
                            st.session_state.current_query = hist['query']
                            st.experimental_rerun()
            else:
                st.info("No queries run in this session yet.")
            
            # Saved Queries
            st.subheader("💾 Saved Queries")
            query_name = st.text_input("Query Name to Save/Delete", key="sql_save_name")
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                if st.button("Save Query", use_container_width=True):
                    if query_name:
                        st.session_state.saved_sql_queries[query_name] = st.session_state.current_query
                        st.success(f"Query '{query_name}' saved!")
                    else:
                        st.warning("Please enter a name to save the query.")
            with s_col2:
                if st.button("Delete Query", use_container_width=True):
                    if query_name and query_name in st.session_state.saved_sql_queries:
                        del st.session_state.saved_sql_queries[query_name]
                        st.success(f"Query '{query_name}' deleted!")
                        st.experimental_rerun()
                    else:
                        st.warning("Enter the name of an existing query to delete.")

            if st.session_state.saved_sql_queries:
                saved_query_to_load = st.selectbox("Load a saved query", options=[""] + list(st.session_state.saved_sql_queries.keys()))
                if saved_query_to_load:
                    st.session_state.current_query = st.session_state.saved_sql_queries[saved_query_to_load]
                    st.info(f"Query '{saved_query_to_load}' loaded into editor.")
                    st.experimental_rerun()

        with main_col:
            st.subheader("✍️ SQL Query Editor")
            st.info("You can run `SELECT` statements to query data, or DML statements like `UPDATE`, `INSERT`, `DELETE` to modify the in-memory DataFrame for this session.")
            query = st.text_area(
                "Enter your SQL query here. The table is named `data`.",
                value=st.session_state.get('current_query', "SELECT * FROM data WHERE age > :min_age;"),
                height=200,
                key="sql_query_editor"
            )
            st.session_state.current_query = query

            # --- Parameter Substitution ---
            final_query = query
            try:
                for param in st.session_state.sql_params:
                    param_name = param.get("Parameter")
                    param_value = param.get("Value")
                    if param_name and param_value is not None:
                        # Basic substitution: add quotes for non-numeric values and escape single quotes
                        if not str(param_value).replace('.', '', 1).isdigit():
                            param_value_safe = str(param_value).replace("'", "''")
                            final_query = final_query.replace(param_name, f"'{param_value_safe}'")
                        else:
                            final_query = final_query.replace(param_name, str(param_value))
                
                if final_query != query:
                    with st.expander("Substituted Query Preview"):
                        st.code(final_query, language='sql')
            except Exception as e:
                st.error(f"Error substituting parameters: {e}")

            # --- Action Buttons ---
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                if st.button("🚀 Execute Query", type="primary", use_container_width=True):
                    is_dml = any(final_query.strip().upper().startswith(s) for s in ['INSERT', 'UPDATE', 'DELETE'])
                    
                    result, error, duration = execute_sql_query(df, final_query) # result is new full df for DML
                    st.session_state.sql_query_duration = duration
                    if error:
                        st.session_state.sql_result = None
                        st.error(f"SQL Error: {error}")
                    else:
                        if is_dml:
                            st.session_state.df = result # Update the main dataframe
                            st.session_state.sql_result = None # Clear previous query results
                            st.success(f"DML query executed successfully in {duration:.4f}s. The data has been updated.")
                            st.info("The main DataFrame has been updated. Rerunning to reflect changes across the app.")
                            st.experimental_rerun()
                        else:
                            # This is a SELECT query
                            st.session_state.sql_result = result
                            st.session_state.sql_history.append({
                                'query': query, # Save original query with parameters
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'rows': len(result),
                                'duration': duration
                            })
                            st.success(f"Query executed successfully in {duration:.4f} seconds! Returned {len(result)} rows.")
            with b_col2:
                if st.button("📊 Explain Query", use_container_width=True):
                    if final_query:
                        explain_result, explain_error, _ = execute_sql_query(df, f"EXPLAIN {final_query}")
                        if explain_error:
                            st.error(f"Error explaining query: {explain_error}")
                        else:
                            st.subheader("🔍 Query Execution Plan (from DuckDB)")
                            st.code(explain_result.to_string(), language='text')
            with b_col3:
                if st.button("🧠 Optimize with AI", use_container_width=True):
                    if not st.session_state.gemini_model:
                        st.warning("Enter your Google AI API Key to use this feature.")
                    elif final_query:
                        schema_str = pd.DataFrame({'Column': df.columns, 'DataType': df.dtypes.astype(str)}).to_string()
                        prompt = f"""You are a DuckDB performance tuning expert.
Given a table named `data` with the following schema:
{schema_str}

The user has written the following SQL query:
```sql
{final_query}
```

Please analyze this query and provide an optimized version if possible.
Explain the key optimizations you made and why they improve performance (e.g., filter pushdown, join order, using specific functions).
If the query is already optimal, state that and explain why.
Format your response using markdown.
"""
                        optimization_suggestion = generate_gemini_content(prompt)
                        if optimization_suggestion:
                            st.subheader("🤖 AI Optimization Suggestion")
                            st.markdown(optimization_suggestion)
            # The 'Format Query' button functionality has been removed as per the request
            # to remove dependency on 'sql_formatter.api'.
            # The b_col4 column is now empty.
            with b_col4: pass

            # Display Results
            if 'sql_result' in st.session_state and st.session_state.sql_result is not None:
                st.subheader("📊 Query Results")
                result_df = st.session_state.sql_result
                st.dataframe(result_df)
                
                # --- AI Interpretation Button ---
                if not result_df.empty:
                    if st.button("🤖 Interpret Results with AI"):
                        if not st.session_state.gemini_model:
                            st.warning("Enter your Google AI API Key to use this feature.")
                        else:
                            prompt = f"""You are a data analyst.
The following SQL query was run:
```sql
{final_query}
```
It produced the following result (showing the first 10 rows):
```
{result_df.head(10).to_string()}
```
The result has a total of {len(result_df)} rows.

Please provide a brief, insightful interpretation of what these results mean.
Focus on the business or analytical implications of the findings.
"""
                            interpretation = generate_gemini_content(prompt)
                            if interpretation:
                                st.subheader("🤖 AI Result Interpretation")
                                st.markdown(interpretation)

                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "💾 Download CSV",
                        result_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"sql_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        "📄 Download JSON",
                        result_df.to_json(orient='records'),
                        file_name=f"sql_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

                # Visualization of results
                with st.expander("🎨 Visualize Query Results", expanded=True):
                    if not result_df.empty:
                        st.markdown("Create a quick plot from your query results.")
                        
                        # Get available columns for plotting
                        numeric_cols_res = result_df.select_dtypes(include=np.number).columns.tolist()
                        all_cols_res = result_df.columns.tolist()
                        
                        if not numeric_cols_res:
                            st.info("No numeric columns in the result to plot.")
                        else:
                            plot_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Histogram"], key="sql_viz_type")
                            
                            try:
                                if plot_type in ["Bar", "Line", "Scatter"]:
                                    viz_cols = st.columns(2)
                                    with viz_cols[0]:
                                        x_axis = st.selectbox("X-axis", all_cols_res, key="sql_viz_x")
                                    with viz_cols[1]:
                                        y_axis = st.selectbox("Y-axis", numeric_cols_res, key="sql_viz_y")
                                    
                                    if x_axis and y_axis:
                                        if plot_type == "Bar":
                                            fig = px.bar(result_df, x=x_axis, y=y_axis, title=f"Bar Chart: {y_axis} by {x_axis}")
                                        elif plot_type == "Line":
                                            fig = px.line(result_df, x=x_axis, y=y_axis, title=f"Line Chart: {y_axis} over {x_axis}")
                                        elif plot_type == "Scatter":
                                            fig = px.scatter(result_df, x=x_axis, y=y_axis, title=f"Scatter Plot: {y_axis} vs {x_axis}")
                                        st.plotly_chart(fig, use_container_width=True)

                                elif plot_type == "Histogram":
                                    hist_col = st.selectbox("Column for Histogram", numeric_cols_res, key="sql_viz_hist_col")
                                    if hist_col:
                                        fig = px.histogram(result_df, x=hist_col, title=f"Histogram of {hist_col}")
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Could not generate plot: {e}")
                    else:
                        st.info("Result is empty, nothing to visualize.")


elif selected_tool == "📊 Exploratory Data Analysis (EDA)":
    st.markdown('<h2 class="tool-header">📊 Advanced Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.df
        
        # EDA Tools Selection
        eda_tools = [
            "🔍 Data Overview",
            "📈 Distribution Analysis",
            "🔗 Correlation Analysis",
            "🎯 Outlier Detection",
            "📊 Missing Value Analysis",
            "🏷️ Categorical Analysis",
            "⏰ Time Series Analysis",
            "🧬 Multivariate Analysis",
            "🎨 Advanced Charting Studio",
            "🔢 Statistical Summary & Tests",
            "⚙️ Dimensionality Reduction",
            "🧩 Clustering Insights",
            "📝 Text Analysis Utilities",
            "🌍 Geospatial Analysis (Basic)",
            "📋 Data Quality Report",
            "🧮 Feature Engineering"
        ]
        
        selected_eda = st.selectbox("Select EDA Tool", eda_tools)
        
        if selected_eda == "🔍 Data Overview":
            # --- Data Overview ---
            st.markdown("### 📜 Dataset Overview & Initial Checks")
            st.markdown("Basic information about the shape, data types, and memory usage of your dataset.")

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Dataset Shape")
                st.info(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
                
                st.subheader("🏷️ Data Types")
                dtype_counts = df.dtypes.value_counts()
                # Convert dtype index to strings and values to float for serialization
                fig = px.pie(
                    values=dtype_counts.values.astype(float),
                    names=[str(x) for x in dtype_counts.index],
                    title="Data Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("💾 Memory Usage")
                memory_usage = df.memory_usage(deep=True)
                memory_df = pd.DataFrame({
                    'Column': memory_usage.index,
                    'Memory (MB)': memory_usage.values / 1024**2
                }).sort_values('Memory (MB)', ascending=False)
                
                fig = px.bar(memory_df.head(10), x='Memory (MB)', y='Column', orientation='h',
                           title="Top 10 Columns by Memory Usage")
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_eda == "📈 Distribution Analysis":
            st.markdown("### 📊 Distribution Analysis Dashboard")
            st.markdown("An interactive dashboard to explore the distribution, central tendency, and spread of a numeric column.")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns found for this analysis.")
            else:
                selected_col = st.selectbox("Select a Numeric Column to Analyze", numeric_cols)
                
                col_data = df[selected_col].dropna()

                # --- Top Row: Key Metrics ---
                st.markdown("#### Key Statistical Metrics")
                metric_cols = st.columns(5)
                mean_val = col_data.mean()
                median_val = col_data.median()
                std_val = col_data.std()
                skew_val = col_data.skew()
                kurt_val = col_data.kurtosis()
                metric_cols[0].metric("Mean", f"{mean_val:.2f}")
                metric_cols[1].metric("Median", f"{median_val:.2f}")
                metric_cols[2].metric("Std. Dev.", f"{std_val:.2f}")
                metric_cols[3].metric("Skewness", f"{skew_val:.2f}")
                metric_cols[4].metric("Kurtosis", f"{kurt_val:.2f}")
                
                st.markdown("---")

                # --- Main Dashboard Layout ---
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### Distribution (Histogram & KDE)")
                    # Histogram with KDE
                    fig_hist = px.histogram(col_data, x=selected_col, nbins=50, title=f"Distribution of {selected_col}", marginal="rug", opacity=0.7, histnorm='probability density')
                    # Add KDE trace using seaborn and plotly
                    kde_data = sns.kdeplot(col_data).get_lines()[0].get_data()
                    plt.clf() # clear seaborn plot
                    fig_hist.add_trace(go.Scatter(x=kde_data[0], y=kde_data[1], mode='lines', name='KDE', line=dict(color='red', width=2)))
                    st.plotly_chart(fig_hist, use_container_width=True)

                    st.markdown("##### Box & Violin Plot")
                    fig_box = px.violin(col_data, y=selected_col, title=f"Violin and Box Plot of {selected_col}", box=True, points="all")
                    st.plotly_chart(fig_box, use_container_width=True)

                with col2:
                    st.markdown("##### Normality Analysis (QQ Plot)")
                    # QQ Plot
                    fig_qq = go.Figure()
                    qq_data = stats.probplot(col_data, dist="norm", plot=None) # Get data for plot
                    fig_qq.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Ordered Values'))
                    fig_qq.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[1][0]*qq_data[0][0] + qq_data[1][1], mode='lines', name='Fit Line', line=dict(color='red')))
                    fig_qq.update_layout(title=f'QQ Plot for {selected_col}', xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles')
                    st.plotly_chart(fig_qq, use_container_width=True)
                    
                    st.markdown("##### Normality Tests")
                    if len(col_data) >= 3:
                        shapiro_stat, shapiro_p = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                        st.metric("Shapiro-Wilk Test (p-value)", f"{shapiro_p:.4f}", help="Tests if data is from a normal distribution. p > 0.05 suggests normality.")
                        if shapiro_p > 0.05:
                            st.success("The data appears to be normally distributed.")
                        else:
                            st.warning("The data does not appear to be normally distributed.")
                        
                        if st.button("🤖 Get AI Interpretation", key="dist_ai_interp"):
                            if not st.session_state.gemini_model:
                                st.warning("Please configure your Google AI API Key in the sidebar.")
                            else:
                                prompt = f"""
You are an expert data analyst. A user is analyzing the distribution of a numeric column named '{selected_col}'.
Here are the key statistics:
- Mean: {mean_val:.4f}
- Median: {median_val:.4f}
- Standard Deviation: {std_val:.4f}
- Skewness: {skew_val:.4f}
- Kurtosis: {kurt_val:.4f}

A Shapiro-Wilk test for normality resulted in a p-value of {shapiro_p:.6f}.

Based on these statistics and the p-value, provide a concise interpretation of the column's distribution.
Address the following points in your interpretation:
1.  **Central Tendency**: Compare the mean and median. What does this suggest about the distribution's symmetry?
2.  **Shape (Skewness & Kurtosis)**: Explain what the skewness value indicates (e.g., left-skewed, right-skewed, symmetric). Explain what the kurtosis value suggests about the tails and peak of the distribution compared to a normal distribution.
3.  **Normality**: Based on the Shapiro-Wilk p-value (where p <= 0.05 typically rejects normality), conclude whether the data is likely to be normally distributed.
4.  **Overall Summary**: Provide a brief, overall summary of the column's characteristics.

Use markdown for formatting.
"""
                                interpretation = generate_gemini_content(prompt)
                                if interpretation:
                                    st.markdown("---")
                                    st.markdown("#### 🤖 AI-Powered Interpretation")
                                    st.markdown(interpretation)

                st.markdown("---")
                st.markdown("#### 🔄 Data Transformations")
                st.info("Apply common transformations to numeric data to address skewness or non-normality. This creates a new column with the transformed data.")
                
                transform_col = st.selectbox("Select Column to Transform", numeric_cols, key="dist_transform_col")
                transform_type = st.selectbox("Select Transformation Type", ["None", "Log (ln)", "Square Root", "Box-Cox", "Yeo-Johnson"], key="dist_transform_type")
                
                if transform_type != "None" and transform_col:
                    new_transformed_col_name = st.text_input("New Column Name for Transformed Data:", value=f"{transform_col}_{transform_type.lower().replace(' ', '_')}_transformed", key="dist_new_transform_col_name")
                    
                    if st.button("Apply Transformation and Visualize", key="dist_apply_transform"):
                        df_transformed = df.copy()
                        original_data = df_transformed[transform_col].dropna()
                        
                        try:
                            if transform_type == "Log (ln)":
                                if (original_data <= 0).any():
                                    st.error("Log transformation requires all values to be positive. Consider adding a small constant or using Box-Cox/Yeo-Johnson.")
                                    transformed_data = pd.Series([]) # Keep empty if error
                                else:
                                    transformed_data = np.log(original_data)
                            elif transform_type == "Square Root":
                                if (original_data < 0).any():
                                    st.error("Square Root transformation requires all values to be non-negative.")
                                    transformed_data = pd.Series([]) # Keep empty if error
                                else:
                                    transformed_data = np.sqrt(original_data)
                            elif transform_type == "Box-Cox":
                                if (original_data <= 0).any():
                                    st.error("Box-Cox transformation requires all values to be positive. Consider adding a small constant or using Yeo-Johnson.")
                                    transformed_data = pd.Series([]) # Keep empty if error
                                else:
                                    transformed_data, lambda_val = stats.boxcox(original_data)
                                    st.info(f"Box-Cox Lambda (λ): {lambda_val:.4f}")
                            elif transform_type == "Yeo-Johnson":
                                transformed_data, lambda_val = stats.yeojohnson(original_data)
                                st.info(f"Yeo-Johnson Lambda (λ): {lambda_val:.4f}")
                            else:
                                transformed_data = pd.Series([]) # Should not happen with "None" check
                            
                            if not transformed_data.empty:
                                # Add transformed data to the main DataFrame copy
                                df_transformed.loc[original_data.index, new_transformed_col_name] = transformed_data
                                st.session_state.df = df_transformed
                                st.success(f"Transformation applied. New column '{new_transformed_col_name}' created.")
                                
                                # Plot original vs transformed distributions
                                fig_transform = make_subplots(rows=1, cols=2, subplot_titles=(f"Original Distribution of {transform_col}", f"Transformed Distribution ({transform_type})"))
                                fig_transform.add_trace(go.Histogram(x=original_data, name='Original', histnorm='probability density'), row=1, col=1)
                                fig_transform.add_trace(go.Histogram(x=transformed_data, name='Transformed', histnorm='probability density'), row=1, col=2)
                                fig_transform.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig_transform, use_container_width=True)

                                st.dataframe(df_transformed[[transform_col, new_transformed_col_name]].head())

                        except Exception as e:
                            st.error(f"Error applying transformation: {e}")

                    else:
                        st.info("Not enough data for Shapiro-Wilk test (requires > 2 samples).")
        
        elif selected_eda == "🔗 Correlation Analysis":
            st.markdown("### 🔗 Correlation Analysis Dashboard")
            st.markdown("Explore relationships between numeric variables using a heatmap and an interactive network graph.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation analysis!")
            else:
                corr_method = st.selectbox(
                    "Select Correlation Method", 
                    ["pearson", "spearman", "kendall"], 
                    help="""
- **Pearson**: Measures linear correlation. Assumes data is normally distributed.
- **Spearman**: Measures rank correlation. Good for non-linear, monotonic relationships.
- **Kendall**: Measures rank correlation. Robust to outliers and smaller sample sizes.
"""
                )
                corr_matrix = df[numeric_cols].corr(method=corr_method)
                
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.markdown(f"##### {corr_method.capitalize()} Correlation Table")
                    st.dataframe(corr_matrix)
                    
                    st.markdown("##### Strongest Correlations")
                    # Unstack the correlation matrix and find the strongest correlations
                    corr_unstacked = corr_matrix.unstack().sort_values(ascending=False)
                    corr_unstacked = corr_unstacked[corr_unstacked != 1.0] # Remove self-correlations
                    strongest_corr = corr_unstacked.drop_duplicates().head(10)
                    st.dataframe(strongest_corr.to_frame(name='Correlation'))

                with col2:
                    st.markdown("##### Correlation Heatmap")
                    fig_heatmap = px.imshow(corr_matrix, 
                                            title="Correlation Matrix of Numeric Columns",
                                            color_continuous_scale="RdBu_r",
                                            zmin=-1, zmax=1,
                                            text_auto=".2f",
                                            aspect="auto")
                    fig_heatmap.update_traces(textfont_size=10)
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                st.markdown("---")
                st.markdown("### 🌐 Advanced: Correlation Network Graph")
                st.info("This graph visualizes correlations as a network. Nodes are variables, and edges represent the strength of the correlation between them. Thicker, brighter lines indicate stronger correlations.")
                
                corr_threshold = st.slider("Correlation Threshold for Network", 0.1, 1.0, 0.5, 0.05)

                # Create network graph data
                nodes = list(corr_matrix.columns)
                edges = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) >= corr_threshold:
                            edges.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
                
                if not edges:
                    st.warning("No correlations found above the selected threshold. Try lowering the threshold.")
                else:
                    # Create node positions (simple circular layout)
                    num_nodes = len(nodes)
                    angle = 2 * np.pi / num_nodes
                    node_x = [np.cos(i * angle) for i in range(num_nodes)]
                    node_y = [np.sin(i * angle) for i in range(num_nodes)]

                    # Create edge traces
                    edge_traces = []
                    for edge in edges:
                        node1, node2, weight = edge
                        x0, y0 = node_x[nodes.index(node1)], node_y[nodes.index(node1)]
                        x1, y1 = node_x[nodes.index(node2)], node_y[nodes.index(node2)]
                        
                        color = 'red' if weight > 0 else 'blue'
                        width = 1 + (abs(weight) - corr_threshold) * 10 # Scale width
                        
                        edge_traces.append(go.Scatter(
                            x=[x0, x1, None], y=[y0, y1, None],
                            line=dict(width=width, color=color),
                            hoverinfo='text',
                            text=f'{node1} - {node2}<br>Corr: {weight:.2f}',
                            mode='lines'))

                    # Create node trace
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=nodes,
                        textposition="bottom center",
                        hoverinfo='text',
                        textfont=dict(size=12),
                        marker=dict(
                            showscale=False,
                            colorscale='YlGnBu',
                            reversescale=True,
                            color=[],
                            size=20,
                            line_width=2))

                    # Create figure
                    fig_network = go.Figure(data=edge_traces + [node_trace],
                                 layout=go.Layout(
                                    title='Correlation Network Graph',
                                    titlefont_size=16,
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                    )
                    st.plotly_chart(fig_network, use_container_width=True)

                # Keep the pairwise scatter plots as an optional, expandable section
                with st.expander("#### Pairwise Scatter Plots for Highly Correlated Variables"):
                    strong_corr_list = [(edge[0], edge[1], edge[2]) for edge in edges if abs(edge[2]) > 0.7]
                    if strong_corr_list:
                        num_pair_plots = st.slider("Number of Pair Plots to Show", 1, min(5, len(strong_corr_list)), min(3, len(strong_corr_list)), key="corr_pair_plots")
                        for i in range(num_pair_plots):
                            var1, var2, corr_val = strong_corr_list[i]
                            fig_pair_scatter = px.scatter(df, x=var1, y=var2, title=f"{var1} vs {var2} (Correlation: {corr_val:.2f})",
                                                          marginal_x="histogram", marginal_y="histogram", trendline="ols")
                            st.plotly_chart(fig_pair_scatter, use_container_width=True)
                    else:
                        st.info("No strong correlations (|r| > 0.7) to generate pair plots.")

        elif selected_eda == "🎯 Outlier Detection":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns found!")
            else:
                selected_col = st.selectbox("Select Column for Outlier Detection", numeric_cols)
                
                outlier_methods = advanced_outlier_detection(df, selected_col)
                
                col1, col2 = st.columns([1,2]) # Adjust column ratio
                
                with col1:
                    st.subheader("🔍 Outlier Detection Results")
                    st.info("Number of outliers detected by different methods.")
                    for method, count in outlier_methods.items():
                        st.metric(f"{method} Outliers", count, help=f"Number of outliers detected by {method} method.")
                
                with col2:
                    st.subheader("📊 Outlier Visualization")
                    viz_method = st.selectbox("Select method to visualize", list(outlier_methods.keys()))

                    fig = go.Figure()
                    col_data = df[[selected_col]].dropna()

                    # Determine outliers based on selected method
                    if viz_method == 'IQR':
                        Q1 = col_data[selected_col].quantile(0.25)
                        Q3 = col_data[selected_col].quantile(0.75)
                        IQR = Q3 - Q1
                        outlier_mask = (col_data[selected_col] < Q1 - 1.5 * IQR) | (col_data[selected_col] > Q3 + 1.5 * IQR)
                    elif viz_method == 'Z-Score':
                        z_scores = np.abs(stats.zscore(col_data[selected_col]))
                        outlier_mask = z_scores > 3
                    elif viz_method == 'Isolation Forest':
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outlier_labels = iso_forest.fit_predict(col_data)
                        outlier_mask = outlier_labels == -1
                    elif viz_method == 'Local Outlier Factor':
                        from sklearn.neighbors import LocalOutlierFactor
                        lof = LocalOutlierFactor(n_neighbors=min(20, len(col_data)-1), contamination='auto')
                        outlier_labels = lof.fit_predict(col_data)
                        outlier_mask = outlier_labels == -1
                    else:
                        outlier_mask = pd.Series([False] * len(col_data), index=col_data.index)

                    # Plot normal points
                    fig.add_trace(go.Scatter(
                        x=col_data.index[~outlier_mask],
                        y=col_data[selected_col][~outlier_mask],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=4)
                    ))
                    
                    # Plot outliers
                    fig.add_trace(go.Scatter(
                        x=col_data.index[outlier_mask],
                        y=col_data[selected_col][outlier_mask],
                        mode='markers',
                        name=f'{viz_method} Outliers',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                    
                    fig.update_layout(title=f"Outlier Visualization for {selected_col} using {viz_method}")
                    st.plotly_chart(fig, use_container_width=True)

        elif selected_eda == "📊 Missing Value Analysis":
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df)) * 100
            
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': missing_percent.values
            }).sort_values('Missing Count', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Missing Value Summary")
                st.dataframe(missing_df[missing_df['Missing Count'] > 0])
            
            with col2:
                if missing_df['Missing Count'].sum() > 0:
                    fig = px.bar(missing_df[missing_df['Missing Count'] > 0].head(10),
                               x='Missing Count', y='Column', orientation='h',
                               title="Missing Values by Column")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No missing values found!")
            
            # Missing value heatmap
            if missing_df['Missing Count'].sum() > 0:
                st.subheader("🔥 Missing Value Heatmap")
                fig = px.imshow(df.isnull().astype(int), 
                              title="Missing Value Pattern",
                              color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_eda == "🏷️ Categorical Analysis":
            st.markdown("### 🏷️ Categorical Analysis Dashboard")
            st.markdown("An interactive dashboard to explore categorical columns.")
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols:
                st.warning("No categorical columns found for this analysis.")
            else:
                selected_col = st.selectbox("Select a Categorical Column to Analyze", categorical_cols)
                
                value_counts = df[selected_col].value_counts()
                top_n = st.slider("Select Top N categories to display", 1, min(50, len(value_counts)), min(10, len(value_counts)), key="cat_top_n")
                value_counts_top_n = value_counts.head(top_n)

                # --- Top Row: Key Metrics ---
                st.markdown("#### Key Categorical Metrics")
                metric_cols = st.columns(4)
                metric_cols[0].metric("Total Categories", f"{len(value_counts):,}")
                metric_cols[1].metric("Most Frequent (Mode)", f"{value_counts.index[0]}")
                metric_cols[2].metric("Mode Frequency", f"{value_counts.iloc[0]:,}")
                metric_cols[3].metric("Mode Percentage", f"{value_counts.iloc[0] / len(df) * 100:.1f}%")

                st.markdown("---")
                
                # --- Main Dashboard Layout ---
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"##### Top {top_n} Categories Visualization")
                    plot_type = st.radio("Select Plot Type", ["Bar Chart", "Donut Chart", "Treemap"], horizontal=True, key="cat_plot_type")
                    
                    if plot_type == "Bar Chart":
                        fig = px.bar(value_counts_top_n, x=value_counts_top_n.index, y=value_counts_top_n.values,
                                     labels={'x': selected_col, 'y': 'Count'}, title=f"Top {top_n} Value Counts for {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    elif plot_type == "Donut Chart":
                        fig = px.pie(value_counts_top_n, names=value_counts_top_n.index, values=value_counts_top_n.values,
                                     title=f"Top {top_n} Value Counts for {selected_col}", hole=0.4)
                        st.plotly_chart(fig, use_container_width=True)
                    elif plot_type == "Treemap":
                        treemap_df = value_counts_top_n.reset_index()
                        treemap_df.columns = ['category', 'count']
                        treemap_df['parent'] = selected_col # A single parent for all categories
                        fig = px.treemap(treemap_df, path=['parent', 'category'], values='count',
                                         title=f"Treemap of Top {top_n} Categories in {selected_col}",
                                         color='count', color_continuous_scale='Blues')
                        fig.update_traces(root_color="lightgrey")
                        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("##### Frequency Table")
                    st.dataframe(value_counts_top_n.to_frame())

                st.markdown("---")
                st.markdown("#### Bivariate Analysis: Cross-Tabulation & Chi-squared Test")
                if len(categorical_cols) >= 2:
                    cat_col_1 = st.selectbox("Select first categorical column for Chi² test", categorical_cols, key="cat_chi2_1")
                    cat_col_2 = st.selectbox("Select second categorical column for Chi² test", [c for c in categorical_cols if c != cat_col_1], key="cat_chi2_2")
                    if cat_col_1 and cat_col_2:
                        contingency_table = pd.crosstab(df[cat_col_1], df[cat_col_2])
                        st.write("Contingency Table:")
                        st.dataframe(contingency_table)
                        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                        st.metric("Chi-squared Statistic", f"{chi2:.2f}")
                        st.metric("P-value", f"{p:.3f}")
                        st.caption("Low p-value (<0.05) suggests a significant association between the two variables.")
                else:
                    st.info("Need at least two categorical columns for Chi-squared test.")
        
        elif selected_eda == "⏰ Time Series Analysis":
            # Detect potential time columns
            time_cols = []
            for col in df.columns:
                if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                    time_cols.append(col)
            
            if not time_cols:
                st.warning("No time columns detected. Try converting columns to datetime first.")
                
                # Option to convert columns
                potential_cols = st.multiselect("Select columns to convert to datetime", df.columns.tolist())
                if potential_cols and st.button("Convert to DateTime"):
                    for col in potential_cols:
                        try:
                            df[col] = pd.to_datetime(df[col])
                            st.success(f"Converted {col} to datetime")
                            time_cols.append(col)
                        except:
                            st.error(f"Could not convert {col} to datetime")
            
            if time_cols:
                time_col = st.selectbox("Select Time Column", time_cols)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    value_col = st.selectbox("Select Value Column", numeric_cols)
                    
                    # Time series plot
                    fig = px.line(df, x=time_col, y=value_col, 
                                title=f"Time Series: {value_col} over {time_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Basic time series statistics
                    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                        time_range = df[time_col].max() - df[time_col].min()
                        st.metric("Time Range", str(time_range))
                        st.metric("Data Points", len(df))
                        st.metric("Average Value", f"{df[value_col].mean():.2f}")                    

                    st.markdown("#### Time Series Decomposition")
                    st.info("Decomposes the time series into trend, seasonality, and residuals using `statsmodels`.")
                    decomp_period = st.number_input("Seasonality Period (e.g., 12 for monthly, 7 for daily)", min_value=2, value=12, step=1)
                    if len(df[value_col].dropna()) >= 2 * decomp_period:
                        try:
                            # Ensure the time column is the index for decomposition
                            ts_data = df.set_index(time_col)[value_col].dropna()
                            result_decompose = seasonal_decompose(ts_data, model='additive', period=decomp_period)
                            
                            fig_decompose = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                                          subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
                            fig_decompose.add_trace(go.Scatter(x=result_decompose.observed.index, y=result_decompose.observed, mode='lines', name='Observed'), row=1, col=1)
                            fig_decompose.add_trace(go.Scatter(x=result_decompose.trend.index, y=result_decompose.trend, mode='lines', name='Trend'), row=2, col=1)
                            fig_decompose.add_trace(go.Scatter(x=result_decompose.seasonal.index, y=result_decompose.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
                            fig_decompose.add_trace(go.Scatter(x=result_decompose.resid.index, y=result_decompose.resid, mode='markers', name='Residual'), row=4, col=1)
                            fig_decompose.update_layout(height=600, title_text="Time Series Decomposition")
                            st.plotly_chart(fig_decompose, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not perform decomposition: {e}")
                    else:
                        st.warning(f"Not enough data points for decomposition with period {decomp_period}. Need at least {2 * decomp_period} points.")

                    st.markdown("#### Autocorrelation (ACF) and Partial Autocorrelation (PACF) Plots")
                    st.info("These plots from `statsmodels` help identify seasonality and lag effects in time series data.")
                    n_lags = st.slider("Number of Lags to Show", 1, 40, 20)
                    if len(df[value_col].dropna()) > n_lags:
                        try:
                            fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
                            plot_acf(df[value_col].dropna(), ax=ax_acf, lags=n_lags)
                            ax_acf.set_title("Autocorrelation (ACF)")
                            st.pyplot(fig_acf)

                            fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
                            plot_pacf(df[value_col].dropna(), ax=ax_pacf, lags=n_lags)
                            ax_pacf.set_title("Partial Autocorrelation (PACF)")
                            st.pyplot(fig_pacf)
                        except Exception as e:
                            st.error(f"Could not generate ACF/PACF plots: {e}")
                    else:
                        st.warning(f"Not enough data points to calculate {n_lags} lags.")

        elif selected_eda == "🔢 Statistical Summary & Tests":
            st.subheader("🔢 Detailed Statistical Summary & Hypothesis Tests")
            
            st.markdown("#### Overall Descriptive Statistics (Numeric Columns)")
            st.dataframe(df.describe(include=[np.number]))

            st.markdown("#### Overall Descriptive Statistics (Object/Categorical Columns)")
            st.dataframe(df.describe(include=['object', 'category']))

            st.markdown("#### Per-Column Statistics")
            for col in df.columns:
                with st.expander(f"Statistics for Column: {col} (Type: {df[col].dtype})"):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        st.write(df[col].describe())
                        st.write(f"**Skewness:** {df[col].skew():.3f}")
                        st.write(f"**Kurtosis:** {df[col].kurtosis():.3f}")
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        st.write(df[col].describe())
                    else: # Categorical or Object
                        st.write(df[col].describe())
                        modes = df[col].mode().tolist()
                        if len(modes) > 10: # If more than 10 modes, truncate
                            st.write(f"**Mode (Top 10):** {modes[:10]} ... (and {len(modes) - 10} more)")
                        else:
                            st.write(f"**Mode:** {modes}")

            st.markdown("#### Hypothesis Testing Utilities")
            numeric_cols_test = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols_test) >= 1:
                st.markdown("##### One-Sample T-test")
                col_ttest1 = st.selectbox("Select column for One-Sample T-test", numeric_cols_test, key="eda_ttest1_col")
                pop_mean = st.number_input("Population Mean (μ₀)", value=0.0, key="eda_ttest1_popmean")
                if st.button("Run One-Sample T-test", key="eda_run_ttest1"):
                    stat, p_val = stats.ttest_1samp(df[col_ttest1].dropna(), pop_mean)
                    st.write(f"T-statistic: {stat:.3f}, P-value: {p_val:.3f}")
                    st.caption("Tests if the mean of a single sample is equal to a known population mean.")

            if len(numeric_cols_test) >= 2:
                st.markdown("##### Two-Sample T-test (Independent)")
                col_ttest2_a = st.selectbox("Select first column for Two-Sample T-test", numeric_cols_test, key="eda_ttest2_col_a")
                col_ttest2_b = st.selectbox("Select second column for Two-Sample T-test", [c for c in numeric_cols_test if c != col_ttest2_a], key="eda_ttest2_col_b")
                if col_ttest2_a and col_ttest2_b and st.button("Run Two-Sample T-test", key="eda_run_ttest2"):
                    stat, p_val = stats.ttest_ind(df[col_ttest2_a].dropna(), df[col_ttest2_b].dropna())
                    st.write(f"T-statistic: {stat:.3f}, P-value: {p_val:.3f}")
                    st.caption("Tests if the means of two independent samples are equal.")

            if len(numeric_cols_test) >= 2 and len(categorical_cols) >=1 :
                st.markdown("##### ANOVA (One-Way)")
                anova_num_col = st.selectbox("Select numeric column for ANOVA", numeric_cols_test, key="eda_anova_num")
                anova_cat_col = st.selectbox("Select categorical column for ANOVA groups", categorical_cols, key="eda_anova_cat")
                if anova_num_col and anova_cat_col:
                    groups = [df[anova_num_col][df[anova_cat_col] == cat].dropna() for cat in df[anova_cat_col].unique() if len(df[anova_num_col][df[anova_cat_col] == cat].dropna()) > 1]
                    if len(groups) >= 2:
                        if st.button("Run ANOVA", key="eda_run_anova"):
                            f_stat, p_val = stats.f_oneway(*groups)
                            st.write(f"F-statistic: {f_stat:.3f}, P-value: {p_val:.3f}")
                            st.caption("Tests if the means of two or more groups are equal.")
                    else:
                        st.warning("Not enough groups with sufficient data for ANOVA.")
            else:
                st.info("ANOVA requires at least two numeric columns and one categorical column with multiple groups.")


        elif selected_eda == "🧬 Multivariate Analysis":
            st.subheader("🧬 Advanced Multivariate Visualizations")
            numeric_cols_mv = df.select_dtypes(include=np.number).columns.tolist()

            if len(numeric_cols_mv) < 2:
                st.warning("Multivariate analysis requires at least two numeric columns.")
            else:
                st.markdown("#### 📊 Pair Plot (Scatter Matrix)")
                st.info("Visualizes pairwise relationships between selected numeric variables. Diagonal shows histograms or KDEs.")
                
                pair_plot_cols = st.multiselect(
                    "Select columns for Pair Plot (2-5 recommended for performance)", 
                    numeric_cols_mv, 
                    default=numeric_cols_mv[:min(len(numeric_cols_mv), 4)], # Default to first 4 or fewer
                    key="mv_pair_plot_cols"
                )
                
                if pair_plot_cols and len(pair_plot_cols) >= 2:
                    hue_col_pair = st.selectbox("Color by (Categorical Column - Optional)", ['None'] + df.select_dtypes(include=['object', 'category']).columns.tolist(), key="mv_pair_plot_hue")
                    hue_col_pair = None if hue_col_pair == 'None' else hue_col_pair

                    if st.button("Generate Pair Plot", key="mv_generate_pair_plot"):
                        with st.spinner("Generating Pair Plot..."):
                            try:
                                fig_pair = px.scatter_matrix(df, dimensions=pair_plot_cols, color=hue_col_pair, title="Pair Plot of Selected Variables")
                                fig_pair.update_layout(height=max(600, 200 * len(pair_plot_cols))) # Adjust height
                                st.plotly_chart(fig_pair, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating pair plot: {e}")
                elif pair_plot_cols and len(pair_plot_cols) < 2:
                    st.warning("Please select at least two columns for the pair plot.")

                st.markdown("---")
                st.markdown("#### 🧊 3D Scatter Plot")
                if len(numeric_cols_mv) < 3:
                    st.info("3D Scatter Plot requires at least three numeric columns.")
                else:
                    x_3d = st.selectbox("Select X-axis for 3D Scatter", numeric_cols_mv, index=0, key="mv_3d_x")
                    y_3d = st.selectbox("Select Y-axis for 3D Scatter", numeric_cols_mv, index=1 if len(numeric_cols_mv) > 1 else 0, key="mv_3d_y")
                    z_3d = st.selectbox("Select Z-axis for 3D Scatter", numeric_cols_mv, index=2 if len(numeric_cols_mv) > 2 else 0, key="mv_3d_z")
                    color_3d = st.selectbox("Color by (Optional)", ['None'] + df.columns.tolist(), key="mv_3d_color")
                    color_3d = None if color_3d == 'None' else color_3d
                    
                    if st.button("Generate 3D Scatter Plot", key="mv_generate_3d_scatter"):
                        with st.spinner("Generating 3D Scatter Plot..."):
                            try:
                                fig_3d = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=color_3d, title=f"3D Scatter Plot: {x_3d} vs {y_3d} vs {z_3d}")
                                st.plotly_chart(fig_3d, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating 3D scatter plot: {e}")
                
                st.markdown("---")
                st.markdown("#### 📊 Parallel Coordinates Plot")
                st.info("Visualizes multiple numeric variables, with each variable represented by a vertical axis. Each data point is a line connecting its values across these axes.")
                if len(numeric_cols_mv) >= 3:
                    parallel_cols = st.multiselect(
                        "Select columns for Parallel Coordinates Plot (3-7 recommended)",
                        numeric_cols_mv,
                        default=numeric_cols_mv[:min(len(numeric_cols_mv), 5)],
                        key="mv_parallel_cols"
                    )
                    parallel_color_col = st.selectbox("Color by (Categorical Column - Optional)", ['None'] + df.select_dtypes(include=['object', 'category']).columns.tolist(), key="mv_parallel_color")
                    parallel_color_col = None if parallel_color_col == 'None' else parallel_color_col

                    if parallel_cols and len(parallel_cols) >=2 and st.button("Generate Parallel Coordinates Plot", key="mv_generate_parallel"):
                        with st.spinner("Generating Parallel Coordinates Plot..."):
                            try:
                                fig_parallel = px.parallel_coordinates(df, dimensions=parallel_cols, color=parallel_color_col, title="Parallel Coordinates Plot")
                                st.plotly_chart(fig_parallel, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating parallel coordinates plot: {e}")
                else:
                    st.warning("Parallel Coordinates Plot requires at least 3 numeric columns.")

            st.markdown("#### Stationarity Tests (ADF & KPSS)")
            st.info("Tests for stationarity are crucial for time series forecasting. A stationary series has constant mean, variance, and autocorrelation over time.")
            
            if numeric_cols and time_col:
                ts_data_for_test = df.set_index(time_col)[value_col].dropna()
                if len(ts_data_for_test) > 10: # ADF/KPSS need sufficient data points
                    try:
                        # ADF Test
                        st.markdown("##### Augmented Dickey-Fuller (ADF) Test")
                        adfuller_result = adfuller(ts_data_for_test)
                        st.write(f"ADF Statistic: {adfuller_result[0]:.4f}")
                        st.write(f"P-value: {adfuller_result[1]:.4f}")
                        st.write("Critical Values:")
                        for key, value in adfuller_result[4].items():
                            st.write(f"  {key}: {value:.4f}")
                        
                        if adfuller_result[1] <= 0.05:
                            st.success("Conclusion: The series is likely stationary (reject null hypothesis of non-stationarity).")
                        else:
                            st.warning("Conclusion: The series is likely non-stationary (fail to reject null hypothesis). Consider differencing.")

                        # KPSS Test
                        st.markdown("##### Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test")
                        kpss_result = kpss(ts_data_for_test, regression='c') # 'c' for constant, 'ct' for constant and trend
                        st.write(f"KPSS Statistic: {kpss_result[0]:.4f}")
                        st.write(f"P-value: {kpss_result[1]:.4f}")
                        st.write("Critical Values:")
                        for key, value in kpss_result[3].items():
                            st.write(f"  {key}: {value:.4f}")
                        
                        if kpss_result[1] <= 0.05:
                            st.warning("Conclusion: The series is likely non-stationary (reject null hypothesis of stationarity).")
                        else:
                            st.success("Conclusion: The series is likely stationary (fail to reject null hypothesis).")

                        st.info("Note: ADF and KPSS tests have opposite null hypotheses. If both agree, the conclusion is strong. If they disagree, further investigation is needed.")

                    except Exception as e:
                        st.error(f"Error performing stationarity tests: {e}")
                else:
                    st.info("Not enough data points for stationarity tests (need > 10).")
            else:
                st.info("Please select a time column and a numeric value column for stationarity tests.")

        elif selected_eda == "🧬 Multivariate Analysis":
            st.subheader("🧬 Advanced Multivariate Visualizations")
            numeric_cols_mv = df.select_dtypes(include=np.number).columns.tolist()

            if len(numeric_cols_mv) < 2:
                st.warning("Multivariate analysis requires at least two numeric columns.")
            else:
                st.markdown("#### 📊 Pair Plot (Scatter Matrix)")
                st.info("Visualizes pairwise relationships between selected numeric variables. Diagonal shows histograms or KDEs.")
                
                pair_plot_cols = st.multiselect(
                    "Select columns for Pair Plot (2-5 recommended for performance)", 
                    numeric_cols_mv, 
                    default=numeric_cols_mv[:min(len(numeric_cols_mv), 4)], # Default to first 4 or fewer
                    key="mv_pair_plot_cols"
                )
                
                if pair_plot_cols and len(pair_plot_cols) >= 2:
                    hue_col_pair = st.selectbox("Color by (Categorical Column - Optional)", ['None'] + df.select_dtypes(include=['object', 'category']).columns.tolist(), key="mv_pair_plot_hue")
                    hue_col_pair = None if hue_col_pair == 'None' else hue_col_pair

                    if st.button("Generate Pair Plot", key="mv_generate_pair_plot"):
                        with st.spinner("Generating Pair Plot..."):
                            try:
                                fig_pair = px.scatter_matrix(df, dimensions=pair_plot_cols, color=hue_col_pair, title="Pair Plot of Selected Variables")
                                fig_pair.update_layout(height=max(600, 200 * len(pair_plot_cols))) # Adjust height
                                st.plotly_chart(fig_pair, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating pair plot: {e}")
                elif pair_plot_cols and len(pair_plot_cols) < 2:
                    st.warning("Please select at least two columns for the pair plot.")

                st.markdown("---")
                st.markdown("#### 🧊 3D Scatter Plot")
                if len(numeric_cols_mv) < 3:
                    st.info("3D Scatter Plot requires at least three numeric columns.")
                else:
                    x_3d = st.selectbox("Select X-axis for 3D Scatter", numeric_cols_mv, index=0, key="mv_3d_x")
                    y_3d = st.selectbox("Select Y-axis for 3D Scatter", numeric_cols_mv, index=1 if len(numeric_cols_mv) > 1 else 0, key="mv_3d_y")
                    z_3d = st.selectbox("Select Z-axis for 3D Scatter", numeric_cols_mv, index=2 if len(numeric_cols_mv) > 2 else 0, key="mv_3d_z")
                    color_3d = st.selectbox("Color by (Optional)", ['None'] + df.columns.tolist(), key="mv_3d_color")
                    color_3d = None if color_3d == 'None' else color_3d
                    
                    if st.button("Generate 3D Scatter Plot", key="mv_generate_3d_scatter"):
                        with st.spinner("Generating 3D Scatter Plot..."):
                            try:
                                fig_3d = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, color=color_3d, title=f"3D Scatter Plot: {x_3d} vs {y_3d} vs {z_3d}")
                                st.plotly_chart(fig_3d, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating 3D scatter plot: {e}")
                
                st.markdown("---")
                st.markdown("#### 📊 Parallel Coordinates Plot")
                st.info("Visualizes multiple numeric variables, with each variable represented by a vertical axis. Each data point is a line connecting its values across these axes.")
                if len(numeric_cols_mv) >= 3:
                    parallel_cols = st.multiselect(
                        "Select columns for Parallel Coordinates Plot (3-7 recommended)",
                        numeric_cols_mv,
                        default=numeric_cols_mv[:min(len(numeric_cols_mv), 5)],
                        key="mv_parallel_cols"
                    )
                    parallel_color_col = st.selectbox("Color by (Categorical Column - Optional)", ['None'] + df.select_dtypes(include=['object', 'category']).columns.tolist(), key="mv_parallel_color")
                    parallel_color_col = None if parallel_color_col == 'None' else parallel_color_col

                    if parallel_cols and len(parallel_cols) >=2 and st.button("Generate Parallel Coordinates Plot", key="mv_generate_parallel"):
                        with st.spinner("Generating Parallel Coordinates Plot..."):
                            try:
                                fig_parallel = px.parallel_coordinates(df, dimensions=parallel_cols, color=parallel_color_col, title="Parallel Coordinates Plot")
                                st.plotly_chart(fig_parallel, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating parallel coordinates plot: {e}")
                else:
                    st.warning("Parallel Coordinates Plot requires at least 3 numeric columns.")

        elif selected_eda == "🎨 Advanced Charting Studio":
            st.subheader("🎨 Advanced Charting Studio")
            st.info("Select a plot type and configure its parameters to create custom visualizations.")

            plot_types = [
                "Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Box Plot",
                "Violin Plot", "Density Heatmap", "Density Contour", "3D Scatter Plot",
                "Pie Chart", "Sunburst Chart", "Treemap", "Funnel Chart",
                "Polar Bar Chart", "Area Plot"
            ]
            selected_plot = st.selectbox("Select Plot Type", plot_types, key="studio_plot_type")

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            all_cols = df.columns.tolist()
            
            st.markdown("#### ⚙️ Plot Configuration")
            
            fig = None

            try:
                if selected_plot == "Scatter Plot":
                    cols = st.columns(4)
                    x_ax = cols[0].selectbox("X-axis (Numeric)", numeric_cols, key="s_x")
                    y_ax = cols[1].selectbox("Y-axis (Numeric)", numeric_cols, index=min(1, len(numeric_cols)-1), key="s_y")
                    color = cols[2].selectbox("Color by (Optional)", ['None'] + all_cols, key="s_c")
                    size = cols[3].selectbox("Size by (Numeric, Optional)", ['None'] + numeric_cols, key="s_s")
                    if x_ax and y_ax:
                        fig = px.scatter(df, x=x_ax, y=y_ax, 
                                         color=None if color == 'None' else color,
                                         size=None if size == 'None' else size,
                                         title=f"Scatter Plot of {x_ax} vs {y_ax}")

                elif selected_plot == "Line Plot":
                    cols = st.columns(3)
                    x_ax = cols[0].selectbox("X-axis", all_cols, key="l_x")
                    y_ax = cols[1].selectbox("Y-axis (Numeric)", numeric_cols, key="l_y")
                    color = cols[2].selectbox("Color by (Optional)", ['None'] + all_cols, key="l_c")
                    if x_ax and y_ax:
                        fig = px.line(df, x=x_ax, y=y_ax, color=None if color == 'None' else color,
                                      title=f"Line Plot of {y_ax} over {x_ax}")

                elif selected_plot == "Bar Chart":
                    cols = st.columns(3)
                    x_ax = cols[0].selectbox("X-axis (Categorical)", categorical_cols, key="b_x")
                    y_ax = cols[1].selectbox("Y-axis (Numeric)", numeric_cols, key="b_y")
                    color = cols[2].selectbox("Color by (Optional)", ['None'] + all_cols, key="b_c")
                    if x_ax and y_ax:
                        agg_func = st.selectbox("Aggregation", ["sum", "mean", "count"], key="b_agg")
                        grouped_df = df.groupby(x_ax, as_index=False)[y_ax].agg(agg_func)
                        fig = px.bar(grouped_df, x=x_ax, y=y_ax, color=None if color == 'None' else color,
                                     title=f"Bar Chart of {agg_func.capitalize()} of {y_ax} by {x_ax}")

                elif selected_plot == "Histogram":
                    cols = st.columns(3)
                    x_ax = cols[0].selectbox("X-axis (Numeric)", numeric_cols, key="h_x")
                    color = cols[1].selectbox("Color by (Categorical, Optional)", ['None'] + categorical_cols, key="h_c")
                    marginal = cols[2].selectbox("Marginal Plot", ['None', 'rug', 'box', 'violin'], key="h_m")
                    if x_ax:
                        fig = px.histogram(df, x=x_ax, color=None if color == 'None' else color,
                                           marginal=None if marginal == 'None' else marginal,
                                           title=f"Histogram of {x_ax}")

                elif selected_plot in ["Box Plot", "Violin Plot"]:
                    cols = st.columns(3)
                    x_ax = cols[0].selectbox("X-axis (Categorical, Optional)", ['None'] + categorical_cols, key="bv_x")
                    y_ax = cols[1].selectbox("Y-axis (Numeric)", numeric_cols, key="bv_y")
                    color = cols[2].selectbox("Color by (Categorical, Optional)", ['None'] + categorical_cols, key="bv_c")
                    if y_ax:
                        plot_func = px.box if selected_plot == "Box Plot" else px.violin
                        fig = plot_func(df, x=None if x_ax == 'None' else x_ax, y=y_ax,
                                     color=None if color == 'None' else color,
                                     title=f"{selected_plot} of {y_ax}")

                elif selected_plot in ["Density Heatmap", "Density Contour"]:
                    cols = st.columns(2)
                    x_ax = cols[0].selectbox("X-axis (Numeric)", numeric_cols, key="d_x")
                    y_ax = cols[1].selectbox("Y-axis (Numeric)", numeric_cols, index=min(1, len(numeric_cols)-1), key="d_y")
                    if x_ax and y_ax:
                        plot_func = px.density_heatmap if selected_plot == "Density Heatmap" else px.density_contour
                        fig = plot_func(df, x=x_ax, y=y_ax, title=f"{selected_plot} of {x_ax} vs {y_ax}")

                elif selected_plot == "3D Scatter Plot":
                    cols = st.columns(4)
                    x_ax = cols[0].selectbox("X-axis (Numeric)", numeric_cols, key="3d_x")
                    y_ax = cols[1].selectbox("Y-axis (Numeric)", numeric_cols, index=min(1, len(numeric_cols)-1), key="3d_y")
                    z_ax = cols[2].selectbox("Z-axis (Numeric)", numeric_cols, index=min(2, len(numeric_cols)-1), key="3d_z")
                    color = cols[3].selectbox("Color by (Optional)", ['None'] + all_cols, key="3d_c")
                    if x_ax and y_ax and z_ax:
                        fig = px.scatter_3d(df, x=x_ax, y=y_ax, z=z_ax, color=None if color == 'None' else color, title=f"3D Scatter Plot")

                elif selected_plot in ["Pie Chart", "Funnel Chart"]:
                    cols = st.columns(2)
                    names_col = cols[0].selectbox("Names/Stages (Categorical)", categorical_cols, key="pf_n")
                    values_col = cols[1].selectbox("Values (Numeric)", numeric_cols, key="pf_v")
                    if names_col and values_col:
                        plot_func = px.pie if selected_plot == "Pie Chart" else px.funnel
                        fig = plot_func(df, names=names_col, values=values_col, title=f"{selected_plot}")

                elif selected_plot in ["Sunburst Chart", "Treemap"]:
                    path = st.multiselect("Hierarchy Path (Categorical)", categorical_cols, key="st_p")
                    values = st.selectbox("Values (Numeric)", numeric_cols, key="st_v")
                    if path and values:
                        plot_func = px.sunburst if selected_plot == "Sunburst Chart" else px.treemap
                        fig = plot_func(df, path=path, values=values, title=f"{selected_plot}")

                elif selected_plot == "Polar Bar Chart":
                    cols = st.columns(3)
                    r = cols[0].selectbox("Radius (Numeric)", numeric_cols, key="pol_r")
                    theta = cols[1].selectbox("Angle (Categorical)", categorical_cols, key="pol_t")
                    color = cols[2].selectbox("Color by (Optional)", ['None'] + all_cols, key="pol_c")
                    if r and theta:
                        fig = px.bar_polar(df, r=r, theta=theta, color=None if color == 'None' else color, title=f"Polar Bar Chart")

                elif selected_plot == "Area Plot":
                    cols = st.columns(3)
                    x_ax = cols[0].selectbox("X-axis", all_cols, key="a_x")
                    y_ax = cols[1].selectbox("Y-axis (Numeric)", numeric_cols, key="a_y")
                    color = cols[2].selectbox("Color by (Categorical, Optional)", ['None'] + categorical_cols, key="a_c")
                    if x_ax and y_ax:
                        fig = px.area(df, x=x_ax, y=y_ax, color=None if color == 'None' else color, title=f"Area Plot of {y_ax} over {x_ax}")
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Configure the plot options above to generate a chart.")

            except Exception as e:
                st.error(f"An error occurred while generating the plot: {e}")
                st.info("Please check your column selections. Some plots have specific requirements (e.g., numeric vs. categorical data).")

        elif selected_eda == "⚙️ Dimensionality Reduction":
            st.subheader("⚙️ Dimensionality Reduction Insights (PCA)")
            numeric_cols_pca = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols_pca) < 2:
                st.warning("PCA requires at least two numeric columns.")
            else:
                st.info("Principal Component Analysis (PCA) is performed on scaled numeric data.")
                
                # Prepare data for PCA
                pca_df = df[numeric_cols_pca].dropna()
                if pca_df.empty or len(pca_df) < 2:
                     st.warning("Not enough data after dropping NaNs for PCA.")
                else:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(pca_df)
                    
                    n_components_pca = st.slider("Number of PCA Components to Analyze", 2, min(10, len(numeric_cols_pca)), min(5, len(numeric_cols_pca)), key="pca_n_components")
                    pca = PCA(n_components=n_components_pca)
                    pca.fit(scaled_data)

                    st.markdown("#### Explained Variance")
                    explained_variance_ratio = pca.explained_variance_ratio_
                    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

                    fig_scree = go.Figure()
                    fig_scree.add_trace(go.Bar(x=list(range(1, n_components_pca + 1)), y=explained_variance_ratio, name='Individual Explained Variance'))
                    fig_scree.add_trace(go.Scatter(x=list(range(1, n_components_pca + 1)), y=cumulative_explained_variance, name='Cumulative Explained Variance', marker_color='red'))
                    fig_scree.update_layout(title='Scree Plot & Cumulative Explained Variance', xaxis_title='Principal Component', yaxis_title='Explained Variance Ratio')
                    st.plotly_chart(fig_scree, use_container_width=True)

                    st.markdown("#### PCA Loadings (Conceptual)")
                    st.info("Loadings show how much each original variable contributes to each principal component. A full loadings plot would be a heatmap or table.")
                    if hasattr(pca, 'components_'):
                        loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components_pca)], index=numeric_cols_pca)
                        st.dataframe(loadings_df.head())

                    st.markdown("#### PCA Projection (2D Scatter)")
                    if n_components_pca >= 2:
                        pca_2d = PCA(n_components=2)
                        projected_data = pca_2d.fit_transform(scaled_data)
                        pca_scatter_df = pd.DataFrame(projected_data, columns=['PC1', 'PC2'])
                        
                        pca_color_col = st.selectbox("Color PCA plot by (Categorical Column - Optional)", ['None'] + df.select_dtypes(include=['object', 'category']).columns.tolist(), key="pca_scatter_color")
                        if pca_color_col != 'None' and pca_color_col in df.columns:
                             # Align color column with pca_df (which had NaNs dropped)
                            pca_scatter_df[pca_color_col] = df.loc[pca_df.index, pca_color_col].values
                        
                        fig_pca_scatter = px.scatter(pca_scatter_df, x='PC1', y='PC2', color=pca_color_col if pca_color_col != 'None' else None, title='2D PCA Projection')
                        st.plotly_chart(fig_pca_scatter, use_container_width=True)


            st.markdown("---")
            st.markdown("#### 🌌 Non-linear Dimensionality Reduction (t-SNE / UMAP)")
            st.info("""
            **t-Distributed Stochastic Neighbor Embedding (t-SNE)** and **Uniform Manifold Approximation and Projection (UMAP)** are powerful non-linear dimensionality reduction techniques.
            Unlike PCA, which focuses on preserving large pairwise distances to maximize variance, t-SNE and UMAP aim to preserve local structures (i.e., relationships between nearby data points).
            This makes them excellent for visualizing high-dimensional data, especially for identifying clusters or manifold structures that linear methods might miss.

            **When to use them:**
            *   **Visualization**: When you want to see if there are natural groupings or patterns in your high-dimensional data that PCA doesn't reveal.
            *   **Exploratory Data Analysis**: To gain insights into the underlying structure of complex datasets.
            *   **Pre-processing for Clustering**: The 2D/3D output can sometimes be a good input for traditional clustering algorithms.

            **Considerations:**
            *   **Computational Cost**: t-SNE can be computationally expensive for very large datasets. UMAP is generally faster.
            *   **Parameter Sensitivity**: Both algorithms have parameters (e.g., `perplexity` for t-SNE, `n_neighbors` for UMAP) that can significantly affect the resulting visualization. Experimentation is often needed.
            *   **Interpretation**: While they reveal structure, the distances in the t-SNE/UMAP plot don't directly correspond to meaningful distances in the original high-dimensional space. Only the relative proximity matters.

            **To use these, you would typically:**
            1.  Install the necessary libraries (`scikit-learn` for t-SNE, `umap-learn` for UMAP).
            2.  Scale your numeric data (e.g., using `StandardScaler`).
            3.  Apply the algorithm (e.g., `TSNE(n_components=2).fit_transform(scaled_data)`).
            4.  Plot the 2D/3D output, often coloring by known categories or cluster labels.
            """)
            
            st.markdown("##### Example Code Structure (Conceptual)")
            st.code("""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE # or from umap import UMAP
import plotly.express as px

# Assuming 'df' is your DataFrame and 'numeric_cols_pca' are your numeric columns

data_for_tsne = df[numeric_cols_pca].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_tsne)

tsne = TSNE(n_components=2, random_state=42, perplexity=30) # Adjust perplexity as needed
tsne_results = tsne.fit_transform(scaled_data)

tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
# If you have original index or labels, add them back:
# tsne_df['Original_Index'] = data_for_tsne.index
# tsne_df['Category'] = df.loc[data_for_tsne.index, 'Your_Category_Column']

fig_tsne = px.scatter(tsne_df, x='TSNE1', y='TSNE2', 
                      # color='Category', # Uncomment if you have a category column
                      title='t-SNE Projection of Data')
fig_tsne.show() # In Streamlit, you'd use st.plotly_chart(fig_tsne)
            """, language="python")

        elif selected_eda == "📋 Data Quality Report":
            st.subheader("📋 Data Quality Report")
            st.markdown(f"**Total Rows:** {len(df)}")
            st.markdown(f"**Total Columns:** {len(df.columns)}")
            st.markdown(f"**Duplicate Rows:** {df.duplicated().sum()}")
            st.markdown(f"**Total Missing Values:** {df.isnull().sum().sum()}")
            st.markdown(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            # Further details can be linked from other EDA sections (Missing Values, Outliers)
            
            st.markdown("#### Data Type Consistency")
            mixed_type_cols = []
            for col in df.columns:
                if df[col].apply(type).nunique() > 1:
                    mixed_type_cols.append(col)
            if mixed_type_cols:
                st.warning(f"Columns with mixed data types found: {', '.join(mixed_type_cols)}. This can cause issues in analysis.")
            else:
                st.success("No columns with overtly mixed Python data types detected (Pandas dtypes are generally consistent).")

            st.markdown("#### Cardinality of Categorical Features")
            cat_cols_card = df.select_dtypes(include=['object', 'category']).columns
            if not cat_cols_card.empty:
                cardinality_data = []
                for col in cat_cols_card:
                    nunique = df[col].nunique()
                    cardinality_data.append({'Column': col, 'Unique Values': nunique, 'Cardinality Ratio': nunique/len(df)})
                cardinality_df = pd.DataFrame(cardinality_data).sort_values(by="Unique Values", ascending=False)
                st.dataframe(cardinality_df)
                high_cardinality_cols = cardinality_df[cardinality_df['Cardinality Ratio'] > 0.5]['Column'].tolist()
                if high_cardinality_cols:
                    st.warning(f"High cardinality categorical features: {', '.join(high_cardinality_cols)}. Consider encoding or feature engineering.")
            else:
                st.info("No categorical columns to analyze for cardinality.")

        elif selected_eda == "🧩 Clustering Insights":
            st.subheader("🧩 Unsupervised Clustering Insights (K-Means)")
            numeric_cols_cluster = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols_cluster) < 2:
                st.warning("Clustering requires at least two numeric columns.")
            else:
                cluster_df = df[numeric_cols_cluster].dropna()
                if cluster_df.empty or len(cluster_df) < 5: # K-Means needs some data
                    st.warning("Not enough data after dropping NaNs for clustering.")
                else:
                    scaler = StandardScaler()
                    scaled_cluster_data = scaler.fit_transform(cluster_df)

                    st.markdown("#### Elbow Method for Optimal K")
                    inertia = []
                    k_range = range(1, 11)
                    for k_val in k_range:
                        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                        kmeans.fit(scaled_cluster_data)
                        inertia.append(kmeans.inertia_)
                    fig_elbow = px.line(x=list(k_range), y=inertia, title="Elbow Method for K-Means", markers=True, labels={'x':'Number of Clusters (K)', 'y':'Inertia'})
                    st.plotly_chart(fig_elbow, use_container_width=True)
                    st.caption("Look for the 'elbow' point where adding more clusters doesn't significantly reduce inertia.")

                    st.markdown("#### K-Means Clustering Visualization")
                    k_optimal = st.number_input("Select the number of clusters (K) based on the elbow plot:", min_value=2, max_value=15, value=3, step=1)
                    
                    if st.button("Run K-Means and Visualize"):
                        with st.spinner("Running K-Means..."):
                            # Perform K-Means
                            kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
                            cluster_df['cluster'] = kmeans.fit_predict(scaled_cluster_data)

                            # Reduce dimensionality to 2D for visualization
                            pca = PCA(n_components=2)
                            components = pca.fit_transform(scaled_cluster_data)
                            cluster_df['pca1'] = components[:, 0]
                            cluster_df['pca2'] = components[:, 1]
                            
                            # Visualize
                            # Calculate Silhouette Score
                            if len(np.unique(cluster_df['cluster'])) > 1 and len(cluster_df) > 1:
                                silhouette_avg = silhouette_score(scaled_cluster_data, cluster_df['cluster'])
                                st.metric("Silhouette Score", f"{silhouette_avg:.3f}", help="Higher score indicates better-defined clusters.")
                            fig_cluster = px.scatter(cluster_df, x='pca1', y='pca2', color='cluster', color_continuous_scale=px.colors.qualitative.Vivid, title=f'K-Means Clustering (K={k_optimal}) on 2D PCA Projection', labels={'pca1': 'Principal Component 1', 'pca2': 'Principal Component 2'})
                            st.plotly_chart(fig_cluster, use_container_width=True)

                            st.markdown("#### Cluster Profiles")
                            st.info("Showing the mean of numeric features for each cluster.")
                            st.dataframe(cluster_df.groupby('cluster')[numeric_cols_cluster].mean())

        elif selected_eda == "📝 Text Analysis Utilities":
            st.subheader("📝 Text Column Analysis")
            st.info("Analyze text-based columns to find word frequencies and patterns. Requires `wordcloud` library.")
            text_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not text_cols:
                st.warning("No text (object or category type) columns found for analysis.")
            else:
                text_col = st.selectbox("Select a Text Column to Analyze", text_cols)
                
                if text_col:
                    text_series = df[text_col].dropna().astype(str)
                    
                    st.markdown("#### Basic Statistics")
                    stats_cols = st.columns(4)
                    stats_cols[0].metric("Total Words", f"{text_series.str.split().str.len().sum():,}")
                    stats_cols[1].metric("Total Characters", f"{text_series.str.len().sum():,}")
                    
                    all_words = ' '.join(text_series).split()
                    if all_words:
                        avg_word_len = np.mean([len(word) for word in all_words])
                        stats_cols[2].metric("Avg. Word Length", f"{avg_word_len:.2f}")
                    else:
                        stats_cols[2].metric("Avg. Word Length", "N/A")

                    stats_cols[3].metric("Avg. Words per Entry", f"{text_series.str.split().str.len().mean():.2f}")

                    st.markdown("---")
                    st.markdown("#### Word Frequency Analysis")
                    
                    text_corpus = " ".join(text for text in text_series)
                    
                    if text_corpus.strip():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("##### Word Cloud")
                            try:
                                wordcloud = WordCloud(width=400, height=300, background_color='white', colormap='viridis').generate(text_corpus)
                                fig_wc, ax_wc = plt.subplots()
                                ax_wc.imshow(wordcloud, interpolation='bilinear')
                                ax_wc.axis('off')
                                st.pyplot(fig_wc)
                            except Exception as e:
                                st.error(f"Could not generate word cloud: {e}")

                        with col2:
                            st.markdown("##### Top 20 Most Frequent Words")
                            from collections import Counter
                            words = [word for word in re.findall(r'\b\w+\b', text_corpus.lower())]
                            word_counts = Counter(words)
                            top_words_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
                            fig_freq = px.bar(top_words_df, x='Count', y='Word', orientation='h')
                            fig_freq.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                            st.plotly_chart(fig_freq, use_container_width=True)
                    else:
                        st.info("The selected column contains no text to analyze.")

                    st.markdown("#### N-gram Analysis")
                    st.info("N-grams are contiguous sequences of N items (words) from a given sample of text. They are useful for understanding common phrases.")
                    
                    n_gram_value = st.slider("Select N for N-grams", 2, 3, 2) # Bigrams and Trigrams
                    top_n_grams = st.slider("Top N N-grams to display", 5, 50, 20)

                    if st.button(f"Generate Top {top_n_grams} {n_gram_value}-grams"):
                        if text_corpus.strip():
                            try:
                                words = re.findall(r'\b\w+\b', text_corpus.lower())
                                if len(words) < n_gram_value:
                                    st.warning(f"Not enough words to form {n_gram_value}-grams.")
                                else:
                                    n_grams = [' '.join(words[i:i+n_gram_value]) for i in range(len(words) - n_gram_value + 1)]
                                    n_gram_counts = Counter(n_grams)
                                    top_n_grams_df = pd.DataFrame(n_gram_counts.most_common(top_n_grams), columns=[f'{n_gram_value}-gram', 'Count'])
                                    
                                    fig_ngram = px.bar(top_n_grams_df, x='Count', y=f'{n_gram_value}-gram', orientation='h',
                                                       title=f"Top {top_n_grams} {n_gram_value}-grams")
                                    fig_ngram.update_layout(yaxis={'categoryorder':'total ascending'}, height=min(600, top_n_grams * 25 + 100))
                                    st.plotly_chart(fig_ngram, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error generating N-grams: {e}")
                        else:
                            st.info("No text to analyze for N-grams.")

        elif selected_eda == "🌍 Geospatial Analysis (Basic)":
            st.subheader("🌍 Geospatial Visualization")
            st.info("This tool requires columns with latitude and longitude data.")

            lat_col_guess = next((c for c in df.columns if 'lat' in c.lower()), None)
            lon_col_guess = next((c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()), None)
            
            geo_cols = st.columns(2)
            with geo_cols[0]:
                lat_col = st.selectbox("Select Latitude Column", df.columns, index=df.columns.get_loc(lat_col_guess) if lat_col_guess and lat_col_guess in df.columns else 0)
            with geo_cols[1]:
                lon_col = st.selectbox("Select Longitude Column", df.columns, index=df.columns.get_loc(lon_col_guess) if lon_col_guess and lon_col_guess in df.columns else 1)

            st.markdown("#### Map Configuration")
            map_config_cols = st.columns(3)
            with map_config_cols[0]:
                color_col = st.selectbox("Color points by (Optional)", ['None'] + df.columns.tolist(), key="geo_color")
            with map_config_cols[1]:
                size_col = st.selectbox("Size points by (Numeric, Optional)", ['None'] + df.select_dtypes(include=np.number).columns.tolist(), key="geo_size")
            with map_config_cols[2]:
                mapbox_style = st.selectbox("Map Style", ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner"], key="geo_style")
            
            if st.button("Generate Map", type="primary"):
                try:
                    map_df = df.dropna(subset=[lat_col, lon_col])
                    fig_map = px.scatter_mapbox(map_df, lat=lat_col, lon=lon_col, color=None if color_col == 'None' else color_col, size=None if size_col == 'None' else size_col, mapbox_style=mapbox_style, zoom=1, title="Geospatial Data Distribution")
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate map: {e}")

        elif selected_eda == "🧮 Feature Engineering":
            st.subheader("🧮 Feature Engineering")
            
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                fe_type = st.selectbox("Select Feature Engineering Technique", [
                    "Select a technique...",
                    "One-Hot Encode Categorical",
                    "Label Encode Categorical",
                    "Scale Numeric (StandardScaler)",
                    "Bin Numeric Column",
                    "Extract Date/Time Features",
                    "Create Polynomial Features",
                    "Create Interaction Features",
                    "Log/Power Transformation"
                ], key="fe_technique_select")

                if fe_type == "One-Hot Encode Categorical":
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    if not categorical_cols:
                        st.warning("No categorical columns found for One-Hot Encoding.")
                    else:
                        col_to_encode = st.selectbox("Select Column to One-Hot Encode", categorical_cols, key="fe_ohe_col")
                        prefix = st.text_input("Prefix for new columns (optional):", value=col_to_encode, key="fe_ohe_prefix")
                        if st.button("Apply One-Hot Encoding", key="fe_execute_ohe"):
                            try:
                                # Drop original column to avoid redundancy
                                df_processed = pd.get_dummies(df, columns=[col_to_encode], prefix=prefix, dummy_na=False)
                                st.success(f"Applied One-Hot Encoding to '{col_to_encode}'. New shape: {df_processed.shape}")
                                st.dataframe(df_processed.head())
                                st.session_state.df = df_processed # Update main DataFrame
                            except Exception as e:
                                st.error(f"Error applying One-Hot Encoding: {str(e)}")

                elif fe_type == "Label Encode Categorical":
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    if not categorical_cols:
                        st.warning("No categorical columns found for Label Encoding.")
                    else:
                        col_to_encode = st.selectbox("Select Column to Label Encode", categorical_cols, key="fe_label_col")
                        new_col_name = st.text_input("New Column Name (optional, default is original_encoded):", value=f"{col_to_encode}_encoded", key="fe_label_new_col")
                        if st.button("Apply Label Encoding", key="fe_execute_label"):
                            try:
                                le = LabelEncoder()
                                # Handle potential NaNs before encoding
                                if df[col_to_encode].isnull().any():
                                    st.warning(f"Column '{col_to_encode}' contains missing values. LabelEncoder does not handle NaNs by default. NaNs will remain NaN in the new column.")
                                
                                # Create a copy to avoid SettingWithCopyWarning
                                df_processed = df.copy()
                                df_processed[new_col_name if new_col_name else f"{col_to_encode}_encoded"] = le.fit_transform(df_processed[col_to_encode].astype(str)) # Encode as string to handle NaNs and non-string types gracefully
                                st.success(f"Applied Label Encoding to '{col_to_encode}'. Created '{new_col_name if new_col_name else f'{col_to_encode}_encoded'}'.")
                                st.dataframe(df_processed[[col_to_encode, new_col_name if new_col_name else f"{col_to_encode}_encoded"]].head())
                                st.session_state.df = df_processed # Update main DataFrame
                            except Exception as e:
                                st.error(f"Error applying Label Encoding: {str(e)}")

                elif fe_type == "Scale Numeric (StandardScaler)":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_cols:
                        st.warning("No numeric columns found for Scaling.")
                    else:
                        cols_to_scale = st.multiselect("Select Column(s) to Scale", numeric_cols, key="fe_scale_cols")
                        prefix = st.text_input("Prefix for new columns (optional):", value="scaled", key="fe_scale_prefix")
                        if st.button("Apply StandardScaler", key="fe_execute_scale"):
                            if not cols_to_scale:
                                st.warning("Please select at least one column to scale.")
                            else:
                                try:
                                    scaler = StandardScaler()
                                    # Create a copy to avoid SettingWithCopyWarning
                                    df_processed = df.copy()
                                    
                                    scaled_data = scaler.fit_transform(df_processed[cols_to_scale])
                                    
                                    # Create new column names
                                    new_col_names = [f"{prefix}_{col}" for col in cols_to_scale]
                                    
                                    # Add scaled data to the DataFrame
                                    df_processed[new_col_names] = scaled_data
                                    
                                    st.success(f"Applied StandardScaler to {len(cols_to_scale)} column(s).")
                                    st.dataframe(df_processed[[*cols_to_scale, *new_col_names]].head())
                                    st.session_state.df = df_processed # Update main DataFrame
                                except Exception as e:
                                    st.error(f"Error applying StandardScaler: {str(e)}")

                elif fe_type == "Bin Numeric Column":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_cols:
                        st.warning("No numeric columns found for Binning.")
                    else:
                        col_to_bin = st.selectbox("Select Column to Bin", numeric_cols, key="fe_bin_col")
                        num_bins = st.number_input("Number of Bins:", min_value=2, value=10, step=1, key="fe_bin_num")
                        new_col_name = st.text_input("New Column Name:", value=f"{col_to_bin}_binned", key="fe_bin_new_col")
                        if st.button("Apply Binning", key="fe_execute_bin"):
                            if not new_col_name.strip():
                                st.error("Please provide a name for the new column.")
                            else:
                                try:
                                    # Create a copy to avoid SettingWithCopyWarning
                                    df_processed = df.copy()
                                    # Use pd.cut for binning
                                    df_processed[new_col_name] = pd.cut(df_processed[col_to_bin], bins=num_bins, labels=False, include_lowest=True)
                                    st.success(f"Applied Binning to '{col_to_bin}' into {num_bins} bins. Created '{new_col_name}'.")
                                    st.dataframe(df_processed[[col_to_bin, new_col_name]].head())
                                    st.session_state.df = df_processed # Update main DataFrame
                                except Exception as e:
                                    st.error(f"Error applying Binning: {str(e)}")

                elif fe_type == "Extract Date/Time Features":
                    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                    if not datetime_cols:
                        st.warning("No datetime columns found. Please convert a column to datetime first (e.g., in Time Series Analysis).")
                    else:
                        col_to_extract = st.selectbox("Select Datetime Column", datetime_cols, key="fe_datetime_col")
                        features_to_extract = st.multiselect("Select Features to Extract", ["Year", "Month", "Day", "Hour", "Minute", "Second", "Day of Week", "Day of Year", "Week of Year", "Quarter"], key="fe_datetime_features")
                        if st.button("Extract Features", key="fe_execute_datetime"):
                            if not features_to_extract:
                                st.warning("Please select at least one feature to extract.")
                            else:
                                try:
                                    # Create a copy to avoid SettingWithCopyWarning
                                    df_processed = df.copy()
                                    
                                    for feature in features_to_extract:
                                        new_col_name = f"{col_to_extract}_{feature.lower().replace(' ', '_')}"
                                        if feature == "Year":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.year
                                        elif feature == "Month":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.month
                                        elif feature == "Day":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.day
                                        elif feature == "Hour":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.hour
                                        elif feature == "Minute":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.minute
                                        elif feature == "Second":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.second
                                        elif feature == "Day of Week":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.dayofweek # Monday=0, Sunday=6
                                        elif feature == "Day of Year":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.dayofyear
                                        elif feature == "Week of Year":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.isocalendar().week.astype(int) # Use isocalendar for week
                                        elif feature == "Quarter":
                                            df_processed[new_col_name] = df_processed[col_to_extract].dt.quarter
                                    
                                    st.success(f"Extracted {len(features_to_extract)} features from '{col_to_extract}'.")
                                    st.dataframe(df_processed[[col_to_extract] + [f"{col_to_extract}_{f.lower().replace(' ', '_')}" for f in features_to_extract]].head())
                                    st.session_state.df = df_processed # Update main DataFrame
                                except Exception as e:
                                    st.error(f"Error extracting datetime features: {str(e)}")                
                
                elif fe_type == "Create Polynomial Features":
                    numeric_cols_poly = df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_cols_poly:
                        st.warning("No numeric columns for Polynomial Features.")
                    else:
                        poly_cols = st.multiselect("Select numeric columns for Polynomial Features", numeric_cols_poly, key="fe_poly_cols")
                        degree = st.slider("Polynomial Degree", 2, 4, 2, key="fe_poly_degree")
                        interaction_only = st.checkbox("Interaction Terms Only?", value=False, key="fe_poly_interaction")
                        include_bias = st.checkbox("Include Bias Term (intercept)?", value=False, key="fe_poly_bias")

                        if st.button("Generate Polynomial Features", key="fe_execute_poly"):
                            if not poly_cols:
                                st.warning("Please select at least one column.")
                            else:
                                try:
                                    from sklearn.preprocessing import PolynomialFeatures
                                    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
                                    poly_features = poly.fit_transform(df[poly_cols])
                                    poly_feature_names = poly.get_feature_names_out(poly_cols)
                                    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
                                    
                                    # Merge back, avoiding duplicate original columns if include_bias=False and interaction_only=False
                                    df_processed = df.copy()
                                    for col_name in poly_feature_names:
                                        if col_name not in df_processed.columns: # Add only new features
                                            df_processed[col_name] = df_poly[col_name]
                                        elif col_name in poly_cols and (interaction_only or include_bias): # If original col is part of output and we want interactions/bias
                                            df_processed[f"{col_name}_poly"] = df_poly[col_name] # rename to avoid clash if it's just the original

                                    st.success(f"Generated {len(poly_feature_names)} polynomial features. Original columns selected: {len(poly_cols)}")
                                    st.dataframe(df_processed.head())
                                    st.session_state.df = df_processed
                                except Exception as e:
                                    st.error(f"Error generating polynomial features: {e}")
                
                elif fe_type == "Create Interaction Features":
                    numeric_cols_interact = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols_interact) < 2:
                        st.warning("Need at least two numeric columns for Interaction Features.")
                    else:
                        interact_col1 = st.selectbox("Select first numeric column", numeric_cols_interact, key="fe_interact_col1")
                        interact_col2 = st.selectbox("Select second numeric column", [c for c in numeric_cols_interact if c != interact_col1], key="fe_interact_col2")
                        new_col_name_interact = st.text_input("New Interaction Column Name", value=f"{interact_col1}_x_{interact_col2}", key="fe_interact_new_name")
                        if st.button("Create Interaction Feature", key="fe_execute_interact"):
                            if interact_col1 and interact_col2 and new_col_name_interact:
                                try:
                                    df_processed = df.copy()
                                    df_processed[new_col_name_interact] = df_processed[interact_col1] * df_processed[interact_col2]
                                    st.success(f"Created interaction feature '{new_col_name_interact}'.")
                                    st.dataframe(df_processed[[interact_col1, interact_col2, new_col_name_interact]].head())
                                    st.session_state.df = df_processed
                                except Exception as e:
                                    st.error(f"Error creating interaction feature: {e}")
                            else:
                                st.warning("Please select two distinct columns and provide a new column name.")
        
                # Placeholder for Text Analysis and Geospatial Analysis within EDA if relevant columns are detected
                # These would require more specific libraries and checks (e.g., for text: nltk, spacy; for geo: geopandas, folium)


elif selected_tool == "📈 Excel Query Tool":
    st.markdown('<h2 class="tool-header">📈 Super-Advanced Excel Query Tool</h2>', unsafe_allow_html=True)

    # --- Excel Query Examples ---
    with st.expander("📊 Excel Query Examples", expanded=False):
        st.markdown("""
**1. XLOOKUP (New):**
Find the `email` and `city` for the user with `user_id` 123.
- Lookup Value: `123`
- Lookup Column: `user_id`
- Return Columns: `email`, `city`

**2. Advanced Filter (New):**
Show all rows where (`country` is "USA" AND `age` > 30) OR (`status` is "Active").

**3. SUMIFS (New):**
Sum `sales` where `region` is "North" AND `product_category` is "Electronics".

**4. Goal Seek (New):**
Find the universal `discount` (`x`) needed to make the total profit (`(df['price'] - df['cost']) * (1-x)` ) equal to $50,000.

**5. Conditional Formatting (New):**
Highlight all `sales` values over 1000 in green.

**6. PIVOT Example:**
Summarize total `sales` by `region` and `product`.
- Index: `region`
- Columns: `product`
- Values: `sales`
- Aggregation: `sum`
        """)

    if st.session_state.df is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.df
        
        st.subheader("🔧 Excel-Style Operations")
        
        # New categorized structure
        op_category = st.selectbox("Select Operation Category", [
            "Lookup & Reference", "Conditional Aggregation", "Filtering & Sorting", "Pivoting & Grouping",
            "Text Manipulation", "Math & Statistical", "Logical & Data Shaping", "Advanced Tools"
        ], key="excel_op_category")

        operations_map = {
            "Lookup & Reference": ["XLOOKUP (New)", "VLOOKUP", "HLOOKUP", "INDEX/MATCH"],
            "Conditional Aggregation": ["SUMIFS/COUNTIFS/AVERAGEIFS (New)", "SUMIF", "COUNTIF", "AVERAGEIF"],
            "Filtering & Sorting": ["Advanced Filter (New)", "SORT"],
            "Pivoting & Grouping": ["PIVOT", "GROUPBY"],
            "Text Manipulation": ["CONCATENATE", "SPLIT", "Text: LEFT/RIGHT/MID", "Text: FIND/REPLACE", "Text: TRIM/UPPER/LOWER/LEN", "Text: REGEXMATCH (New)", "Text: REGEXREPLACE (New)", "Text: TEXTJOIN (New)"],
            "Math & Statistical": ["Math: ROUND/ABS/SQRT", "Math: POWER/MOD", "Statistical: RANK/PERCENTILE", "Statistical: UNIQUE (New)"],
            "Logical & Data Shaping": ["Logical: IF (Conditional Column)", "Data: Transpose", "Data: Fill Down/Up", "Date/Time: Extract Component", "Data: Custom Column from Row Logic (New)"],
            "Advanced Tools": ["Goal Seek (New)", "Conditional Formatting (New)"]
        }

        operation = st.selectbox("Select Operation", operations_map[op_category], key="excel_operation_select")
        
        if operation == "XLOOKUP (New)":
            st.subheader("🔍 XLOOKUP Operation")
            st.info("A modern and powerful lookup. Finds a value in a column and returns corresponding values from other columns.")
            
            c1, c2 = st.columns(2)
            with c1:
                lookup_value_x = st.text_input("Lookup Value", key="xlookup_value")
                lookup_col_x = st.selectbox("Lookup Column (where to search)", df.columns.tolist(), key="xlookup_col")
                if_not_found_x = st.text_input("Value if not found (optional)", key="xlookup_not_found")
            with c2:
                return_cols_x = st.multiselect("Return Columns (what to get back)", df.columns.tolist(), key="xlookup_return_cols")
                search_direction_x = st.radio("Search Direction", ["First to Last", "Last to First"], key="xlookup_search_dir")

            if st.button("Execute XLOOKUP", key="excel_execute_xlookup"):
                if not lookup_value_x or not lookup_col_x or not return_cols_x:
                    st.warning("Please provide a lookup value, lookup column, and at least one return column.")
                else:
                    try:
                        temp_df = df.copy()
                        # Use a temporary column for matching to handle different types
                        temp_df['_match_col'] = temp_df[lookup_col_x].astype(str)
                        
                        if search_direction_x == "Last to First":
                            temp_df = temp_df.iloc[::-1] # Reverse the DataFrame

                        match_index = temp_df[temp_df['_match_col'] == lookup_value_x].index.values

                        if len(match_index) > 0:
                            result_df = df.loc[match_index[0], return_cols_x]
                            st.success("Match found:")
                            st.dataframe(result_df.to_frame().T)
                        else:
                            st.warning("No exact match found.")
                            if if_not_found_x:
                                st.write(f"Result if not found: {if_not_found_x}")

                    except Exception as e:
                        st.error(f"Error during XLOOKUP: {str(e)}")

        elif operation == "VLOOKUP":
            st.subheader("🔍 VLOOKUP Operation")
            lookup_col = st.selectbox("Lookup Column", df.columns.tolist())
            lookup_value = st.text_input("Lookup Value")
            return_col = st.selectbox("Return Column", df.columns.tolist())
            
            if st.button("Execute VLOOKUP"):
                try:
                    result = df[df[lookup_col].astype(str).str.contains(lookup_value, case=False, na=False)]
                    if not result.empty:
                        st.success(f"Found {len(result)} matches")
                        st.dataframe(result[[lookup_col, return_col]])
                    else:
                        st.warning("No matches found")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif operation == "HLOOKUP":
            st.subheader("↔️ HLOOKUP Operation")
            st.info("HLOOKUP searches for a value in the first selected row of a range and returns the value in the same column from a row you specify.")
            
            # For HLOOKUP, the "table array" is the df itself.
            # We need to select the row to search in (lookup_row_index)
            # And the row to return from (return_row_index)
            
            lookup_row_options = {f"Row {i} (Header is 0)": i for i in range(min(10, len(df)))} # Show first 10 rows as options
            if not df.empty:
                lookup_row_index = st.selectbox("Select Row to Search In (Header is 0, Data starts at 1 if header exists)", list(range(len(df))), format_func=lambda x: f"Row Index {x} (Value: {df.iloc[x,0]}...)", key="hlookup_lookup_row")
                lookup_value_h = st.text_input("Value to Find in Selected Row", key="hlookup_value")
                return_row_index = st.number_input("Row Index to Return Value From (0-based)", min_value=0, max_value=len(df)-1, value=lookup_row_index + 1 if lookup_row_index + 1 < len(df) else lookup_row_index, step=1, key="hlookup_return_row")

                if st.button("Execute HLOOKUP", key="excel_execute_hlookup"):
                    try:
                        search_row_series = df.iloc[lookup_row_index].astype(str)
                        matching_cols = search_row_series[search_row_series.str.contains(lookup_value_h, case=False, na=False)].index
                        if not matching_cols.empty:
                            result_value = df.loc[return_row_index, matching_cols[0]] # Get first match
                            st.success(f"Found '{lookup_value_h}' in column '{matching_cols[0]}' (Row {lookup_row_index}). Value from Row {return_row_index}:")
                            st.write(result_value)
                        else:
                            st.warning(f"Value '{lookup_value_h}' not found in row {lookup_row_index}.")
                    except Exception as e:
                        st.error(f"Error during HLOOKUP: {str(e)}")
            else:
                st.warning("DataFrame is empty.")

        elif operation == "INDEX/MATCH":
            st.subheader("🔍 INDEX/MATCH Operation")
            st.info("Simulates Excel's INDEX/MATCH to find a value in a 'return' column based on matching a 'lookup' value in a 'lookup' column.")
            
            if not df.empty:
                lookup_value_im = st.text_input("Lookup Value", key="im_lookup_value")
                lookup_col_im = st.selectbox("Lookup Column", df.columns.tolist(), key="im_lookup_col")
                return_col_im = st.selectbox("Return Column", df.columns.tolist(), key="im_return_col")
                
                if st.button("Execute INDEX/MATCH", key="excel_execute_index_match"):
                    try:
                        # Find the index of the first match in the lookup column
                        # Use .astype(str) to handle various data types in lookup
                        match_indices = df[lookup_col_im].astype(str).str.contains(lookup_value_im, case=False, na=False)
                        
                        if match_indices.any():
                            # Get the value from the return column at the first matching index
                            result_value = df.loc[match_indices, return_col_im].iloc[0]
                            st.success(f"Found '{lookup_value_im}' in column '{lookup_col_im}'. Corresponding value in '{return_col_im}':")
                            st.write(result_value)
                        else:
                            st.warning(f"Value '{lookup_value_im}' not found in column '{lookup_col_im}'.")
                    except Exception as e:
                        st.error(f"Error during INDEX/MATCH: {str(e)}")
            else:
                st.warning("DataFrame is empty.")
        
        elif operation == "PIVOT":
            st.subheader("📊 Pivot Table")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                index_col = st.selectbox("Index (Rows)", df.columns.tolist())
            with col2:
                columns_col = st.selectbox("Columns", ['None'] + df.columns.tolist())
            with col3:
                values_col = st.selectbox("Values", df.select_dtypes(include=[np.number]).columns.tolist())
            
            agg_func = st.selectbox("Aggregation Function", ['sum', 'mean', 'count', 'min', 'max'])
            
            if st.button("Create Pivot Table"):
                try:
                    if columns_col == 'None':
                        pivot = df.groupby(index_col)[values_col].agg(agg_func).reset_index()
                    else:
                        pivot = df.pivot_table(
                            index=index_col, 
                            columns=columns_col, 
                            values=values_col, 
                            aggfunc=agg_func, 
                            fill_value=0
                        )
                    
                    st.dataframe(pivot)
                    
                    # Visualization
                    if isinstance(pivot, pd.DataFrame) and not pivot.empty:
                        fig = px.bar(pivot.reset_index(), x=pivot.index.name, y=pivot.columns.tolist(),
                                   title=f"Pivot Chart - {agg_func.title()} of {values_col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error creating pivot table: {str(e)}")
        
        elif operation == "Advanced Filter (New)":
            st.subheader("🔍 Advanced Filter with Multiple Conditions")
            st.info("Add multiple rules to filter your data. The first rule is applied directly, subsequent rules are combined using the selected logic (AND/OR).")

            if 'filter_rules' not in st.session_state:
                st.session_state.filter_rules = [{'Logic': 'AND', 'Column': df.columns[0], 'Condition': 'Contains', 'Value': ''}]

            edited_rules = st.data_editor(
                st.session_state.filter_rules,
                num_rows="dynamic",
                column_config={
                    "Logic": st.column_config.SelectboxColumn("Logic", help="How to combine with the previous rule", options=["AND", "OR"], default="AND", required=True),
                    "Column": st.column_config.SelectboxColumn("Column", help="Column to filter", options=df.columns.tolist(), default=df.columns[0], required=True),
                    "Condition": st.column_config.SelectboxColumn("Condition", help="The filter condition", options=["Contains", "Does Not Contain", "Equals", "Not Equal To", "Greater Than", "Less Than", "Is Null", "Is Not Null", "Is in List (comma-separated)", "Is Not in List (comma-separated)"], default="Contains", required=True),
                    "Value": st.column_config.TextColumn("Value", help="Value for the condition (not needed for Is Null/Is Not Null)")
                },
                key="excel_filter_rules_editor"
            )
            st.session_state.filter_rules = edited_rules

            if st.button("Apply Advanced Filter", key="excel_execute_advanced_filter"):
                if not edited_rules or not edited_rules[0]['Column']:
                    st.warning("Please configure at least one filter rule.")
                else:
                    try:
                        final_mask = pd.Series(True, index=df.index)
                        
                        for i, rule in enumerate(edited_rules):
                            col = rule['Column']
                            cond = rule['Condition']
                            val = rule['Value']
                            
                            # Convert value for numeric comparisons
                            numeric_val = pd.to_numeric(val, errors='coerce')

                            if cond == "Is Null":
                                current_mask = df[col].isnull()
                            elif cond == "Is Not Null":
                                current_mask = df[col].notnull()
                            elif cond == "Is in List (comma-separated)":
                                values_list = [v.strip() for v in val.split(',')]
                                current_mask = df[col].astype(str).isin(values_list)
                            elif cond == "Is Not in List (comma-separated)":
                                values_list = [v.strip() for v in val.split(',')]
                                current_mask = ~df[col].astype(str).isin(values_list)
                            # Ensure numeric comparison for numeric conditions
                            elif cond in ["Greater Than", "Less Than"] and not pd.api.types.is_numeric_dtype(df[col]):
                                st.warning(f"Column '{col}' is not numeric. Skipping numeric comparison for rule {i+1}.")
                                continue
                            elif cond == "Contains":
                                current_mask = df[col].astype(str).str.contains(val, case=False, na=False)
                            elif cond == "Does Not Contain":
                                current_mask = ~df[col].astype(str).str.contains(val, case=False, na=False)
                            elif cond == "Equals":
                                current_mask = df[col].astype(str) == val
                            elif cond == "Not Equal To":
                                current_mask = df[col].astype(str) != val
                            elif cond in ["Greater Than", "Less Than"] and not pd.isna(numeric_val):
                                if cond == "Greater Than":
                                    current_mask = df[col] > numeric_val
                                else: # Less Than
                                    current_mask = df[col] < numeric_val
                            else:
                                # If condition is numeric but value is not, or other error, skip rule
                                st.warning(f"Skipping rule {i+1} due to invalid value for numeric comparison.")
                                continue

                            if i == 0:
                                final_mask = current_mask
                            elif rule['Logic'] == 'AND':
                                final_mask = final_mask & current_mask
                            elif rule['Logic'] == 'OR':
                                final_mask = final_mask | current_mask
                        
                        filtered_df = df[final_mask]
                        st.success(f"Filtered to {len(filtered_df)} rows")
                        st.dataframe(filtered_df)

                    except Exception as e:
                        st.error(f"Error applying filter: {e}")
        
        elif operation == "SORT":
            st.subheader("⬆️⬇️ Sort Data")
            sort_col = st.selectbox("Sort Column", df.columns.tolist())
            sort_order = st.radio("Sort Order", ["Ascending", "Descending"])
            
            if st.button("Apply Sort"):
                ascending = sort_order == "Ascending"
                sorted_df = df.sort_values(by=sort_col, ascending=ascending)
                st.dataframe(sorted_df)
        
        elif operation == "GROUPBY":
            st.subheader("📊 Group By & Aggregate")
            
            # Ensure df has columns before proceeding
            if df.empty or len(df.columns) == 0:
                st.warning("DataFrame is empty or has no columns to perform GroupBy.")
            else:
                group_col = st.selectbox("Group By Column", df.columns.tolist(), key="excel_groupby_group_col")
                
                numeric_cols_for_agg = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols_for_agg:
                    st.warning("No numeric columns available for aggregation functions like sum, mean, etc. 'Count' can still be used.")
                    # Allow selection of any column for 'count', or the group_col itself
                    agg_col_options = df.columns.tolist()
                else:
                    agg_col_options = numeric_cols_for_agg
                
                agg_col = st.selectbox("Aggregate Column", agg_col_options, key="excel_groupby_agg_col")
                agg_func = st.selectbox("Aggregation Function", ["sum", "mean", "count", "min", "max", "std"], key="excel_groupby_agg_func")
                
                if st.button("Execute Group By", key="excel_execute_groupby"):
                    try:
                        result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
                        st.dataframe(result)
                        
                        # Visualization
                        fig = px.bar(result, x=group_col, y=agg_col, 
                                   title=f"{agg_func.title()} of {agg_col} by {group_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error executing Group By: {str(e)}")
        
        elif operation == "SUMIFS/COUNTIFS/AVERAGEIFS (New)":
            st.subheader("🧮 Multi-Condition Aggregation")
            agg_type = st.radio("Aggregation Type", ["SUMIFS", "COUNTIFS", "AVERAGEIFS"], horizontal=True, key="excel_aggifs_type")

            if agg_type in ["SUMIFS", "AVERAGEIFS"]:
                agg_col = st.selectbox("Column to Aggregate (must be numeric)", df.select_dtypes(include=[np.number]).columns.tolist(), key="excel_aggifs_agg_col")
            else:
                agg_col = None # Not needed for COUNTIFS

            st.markdown("##### Define Criteria")
            if 'aggifs_rules' not in st.session_state:
                st.session_state.aggifs_rules = [{'Column': df.columns[0], 'Condition': 'Equals', 'Value': ''}]

            edited_rules_aggifs = st.data_editor(
                st.session_state.aggifs_rules,
                num_rows="dynamic",
                column_config={
                    "Column": st.column_config.SelectboxColumn("Criteria Column", options=df.columns.tolist(), required=True),
                    "Condition": st.column_config.SelectboxColumn("Condition", options=["Contains", "Equals", "Greater Than", "Less Than"], required=True),
                    "Value": st.column_config.TextColumn("Criteria Value", required=True)
                },
                key="excel_aggifs_rules_editor"
            )
            st.session_state.aggifs_rules = edited_rules_aggifs

            if st.button(f"Execute {agg_type}", key="excel_execute_aggifs"):
                if (agg_type != "COUNTIFS" and not agg_col):
                    st.error("Please select a numeric column to aggregate for SUMIFS/AVERAGEIFS.")
                else:
                    try:
                        # Start with a mask that is all True
                        final_mask = pd.Series(True, index=df.index)
                        for rule in edited_rules_aggifs:
                            if not rule['Column'] or not rule['Value']: continue
                            col, cond, val = rule['Column'], rule['Condition'], rule['Value']
                            numeric_val = pd.to_numeric(val, errors='coerce')

                            if cond == "Contains":
                                final_mask &= df[col].astype(str).str.contains(val, case=False, na=False)
                            elif cond == "Equals":
                                final_mask &= df[col].astype(str) == val
                            elif cond == "Greater Than" and not pd.isna(numeric_val):
                                final_mask &= df[col] > numeric_val
                            elif cond == "Less Than" and not pd.isna(numeric_val):
                                final_mask &= df[col] < numeric_val
                        
                        if agg_type == "COUNTIFS":
                            result = final_mask.sum()
                        elif agg_type == "SUMIFS":
                            result = df.loc[final_mask, agg_col].sum()
                        elif agg_type == "AVERAGEIFS":
                            result = df.loc[final_mask, agg_col].mean()
                        st.metric(f"Result of {agg_type}", f"{result:,.2f}")
                    except Exception as e:
                        st.error(f"Error executing {agg_type}: {e}")

        elif operation in ["SUMIF", "COUNTIF", "AVERAGEIF"]:
            st.subheader(f"🧮 {operation} Operation")
            condition_col = st.selectbox("Condition Column", df.columns.tolist())
            condition_value = st.text_input("Condition Value")
            
            if operation in ["SUMIF", "AVERAGEIF"]:
                agg_col = st.selectbox("Column to Aggregate", df.select_dtypes(include=[np.number]).columns.tolist())
            
            if st.button(f"Execute {operation}"):
                try:
                    # Create a boolean mask based on the condition
                    # Simple equality check for demonstration
                    mask = df[condition_col].astype(str).str.contains(condition_value, case=False, na=False)
                    
                    if operation == "COUNTIF":
                        result = mask.sum()
                        st.metric(f"Count where {condition_col} contains '{condition_value}'", result)
                    elif operation == "SUMIF":
                        if agg_col:
                            result = df.loc[mask, agg_col].sum()
                            st.metric(f"Sum of {agg_col} where {condition_col} contains '{condition_value}'", f"{result:,.2f}")
                        else:
                            st.warning("Select a numeric column to sum.")
                    elif operation == "AVERAGEIF":
                         if agg_col:
                            result = df.loc[mask, agg_col].mean()
                            st.metric(f"Average of {agg_col} where {condition_col} contains '{condition_value}'", f"{result:,.2f}")
                         else:
                            st.warning("Select a numeric column to average.")
                            
                except Exception as e:
                    st.error(f"Error executing {operation}: {str(e)}")
        
        elif operation == "CONCATENATE":
            st.subheader("📝 Concatenate Columns")
            cols_to_concat = st.multiselect("Select Columns to Concatenate", df.columns.tolist())
            separator = st.text_input("Separator:", value=" ")
            new_col_name = st.text_input("New Column Name:", value="Concatenated_Column")
            
            if cols_to_concat and new_col_name and st.button("Execute Concatenate"):
                try:
                    # Ensure selected columns exist and are not all numeric (which might cause issues with simple .astype(str))
                    if all(col in df.columns for col in cols_to_concat):
                         # Convert all selected columns to string type before concatenating
                        df[new_col_name] = df[cols_to_concat].astype(str).agg(separator.join, axis=1)
                        st.success(f"Created new column '{new_col_name}'")
                        st.dataframe(df[[*cols_to_concat, new_col_name]].head())
                        st.session_state.df = df # Update session state with the new column
                    else:
                        st.error("One or more selected columns not found.")
                except Exception as e:
                    st.error(f"Error concatenating columns: {str(e)}")
        
        elif operation == "SPLIT":
            st.subheader("✂️ Split Column")
            col_to_split = st.selectbox("Select Column to Split", df.columns.tolist())
            delimiter = st.text_input("Delimiter:", value=",")
            new_col_prefix = st.text_input("New Column Prefix:", value=f"{col_to_split}_part")
            max_splits = st.number_input("Max Splits (-1 for all):", min_value=-1, value=-1)
            
            if col_to_split and delimiter and new_col_prefix and st.button("Execute Split"):
                try:
                    # Apply the split operation
                    split_data = df[col_to_split].astype(str).str.split(delimiter, n=max_splits, expand=True)
                    
                    # Generate new column names
                    new_cols = [f"{new_col_prefix}_{i+1}" for i in range(split_data.shape[1])]
                    split_data.columns = new_cols
                    
                    # Join back to the original dataframe
                    df = pd.concat([df, split_data], axis=1)
                    st.success(f"Split column '{col_to_split}' into {split_data.shape[1]} new columns.")
                    st.dataframe(df[[col_to_split, *new_cols]].head())
                    st.session_state.df = df # Update session state
                except Exception as e:
                    st.error("Selected column is not numeric")

        elif operation == "Text: LEFT/RIGHT/MID":
            st.subheader("📝 Text Extraction (LEFT, RIGHT, MID)")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                text_col = st.selectbox("Select Text Column", df.columns.tolist(), key="text_extract_col")
                extract_type = st.radio("Extraction Type", ["LEFT", "RIGHT", "MID"], key="text_extract_type")
                
                if extract_type in ["LEFT", "RIGHT"]:
                    num_chars = st.number_input("Number of Characters", min_value=1, value=5, step=1, key="text_extract_num")
                elif extract_type == "MID":
                    start_num = st.number_input("Start Position (1-based)", min_value=1, value=1, step=1, key="text_extract_start")
                    num_chars = st.number_input("Number of Characters", min_value=1, value=5, step=1, key="text_extract_mid_num")
                
                new_col_name_text_extract = st.text_input("New Column Name:", value=f"{text_col}_{extract_type.lower()}", key="text_extract_new_col")

                if st.button("Execute Text Extraction", key="excel_execute_text_extract"):
                    if not new_col_name_text_extract.strip():
                         st.error("Please provide a name for the new column.")
                    else:
                        try:
                            df_copy = df.copy()
                            # Ensure column is string type and handle NaNs
                            text_series = df_copy[text_col].astype(str).fillna('')
                            
                            if extract_type == "LEFT":
                                df_copy[new_col_name_text_extract] = text_series.str[:num_chars]
                            elif extract_type == "RIGHT":
                                df_copy[new_col_name_text_extract] = text_series.str[-num_chars:]
                            elif extract_type == "MID":
                                # Pandas .str.slice is 0-indexed, end is exclusive
                                df_copy[new_col_name_text_extract] = text_series.str.slice(start=start_num-1, stop=start_num-1 + num_chars)
                            
                            st.success(f"Applied {extract_type} to '{text_col}'. Created '{new_col_name_text_extract}'.")
                            st.dataframe(df_copy[[text_col, new_col_name_text_extract]].head())
                            st.session_state.df = df_copy # Update main DataFrame
                        except Exception as e:
                            st.error(f"Error applying text extraction: {str(e)}")

        elif operation == "Text: FIND/REPLACE":
            st.subheader("📝 Text Find & Replace")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                text_col_fr = st.selectbox("Select Text Column", df.columns.tolist(), key="text_fr_col")
                find_text = st.text_input("Text to Find:", key="text_fr_find")
                replace_with = st.text_input("Replace With:", value="", key="text_fr_replace")
                replace_all = st.checkbox("Replace All occurrences (vs. first)", value=True, key="text_fr_replace_all")
                new_col_name_text_fr = st.text_input("New Column Name (optional, modifies in-place if blank):", key="text_fr_new_col")

                if st.button("Execute Find/Replace", key="excel_execute_text_fr"):
                    if not find_text:
                        st.error("Please enter text to find.")
                    else:
                        try:
                            df_copy = df.copy()
                            # Ensure column is string type and handle NaNs
                            text_series = df_copy[text_col_fr].astype(str).fillna('')
                            
                            if new_col_name_text_fr.strip():
                                df_copy[new_col_name_text_fr] = text_series.str.replace(find_text, replace_with, regex=False, n=-1 if replace_all else 1)
                                st.success(f"Applied Replace to '{text_col_fr}'. Created '{new_col_name_text_fr}'.")
                                st.dataframe(df_copy[[text_col_fr, new_col_name_text_fr]].head())
                                st.session_state.df = df_copy # Update main DataFrame
                            else:
                                # Modify in-place for preview
                                df_copy[text_col_fr] = text_series.str.replace(find_text, replace_with, regex=False, n=-1 if replace_all else 1)
                                st.success(f"Applied Replace to '{text_col_fr}' (in-place preview).")
                                st.dataframe(df_copy[[text_col_fr]].head())
                                # Note: In-place modification in preview mode doesn't update session_state.df

                        except Exception as e:
                            st.error(f"Error applying Find/Replace: {str(e)}")

        elif operation == "Text: TRIM/UPPER/LOWER/LEN":
            st.subheader("📝 Text Formatting (TRIM, UPPER, LOWER, LEN)")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                text_col_format = st.selectbox("Select Text Column", df.columns.tolist(), key="text_format_col")
                format_type = st.radio("Formatting Type", ["TRIM", "UPPER", "LOWER", "LEN"], key="text_format_type")
                new_col_name_text_format = st.text_input("New Column Name (optional, modifies in-place if blank for TRIM/UPPER/LOWER):", key="text_format_new_col")

                if st.button("Execute Text Formatting", key="excel_execute_text_format"):
                    if format_type == "LEN" and not new_col_name_text_format.strip():
                         st.error("Please provide a name for the new column when using LEN.")
                    else:
                        try:
                            df_copy = df.copy()
                            # Ensure column is string type and handle NaNs
                            text_series = df_copy[text_col_format].astype(str).fillna('')
                            
                            if format_type == "TRIM":
                                result_series = text_series.str.strip()
                            elif format_type == "UPPER":
                                result_series = text_series.str.upper()
                            elif format_type == "LOWER":
                                result_series = text_series.str.lower()
                            elif format_type == "LEN":
                                result_series = text_series.str.len()
                            
                            if new_col_name_text_format.strip():
                                df_copy[new_col_name_text_format] = result_series
                                st.success(f"Applied {format_type} to '{text_col_format}'. Created '{new_col_name_text_format}'.")
                                st.dataframe(df_copy[[text_col_format, new_col_name_text_format]].head())
                                st.session_state.df = df_copy # Update main DataFrame
                            else:
                                # Modify in-place for preview (only for TRIM, UPPER, LOWER)
                                if format_type in ["TRIM", "UPPER", "LOWER"]:
                                     df_copy[text_col_format] = result_series
                                     st.success(f"Applied {format_type} to '{text_col_format}' (in-place preview).")
                                     st.dataframe(df_copy[[text_col_format]].head())
                                else: # LEN requires a new column
                                     st.error("LEN operation requires a new column name.")

                        except Exception as e:
                            st.error(f"Error applying text formatting: {str(e)}")

        elif operation == "Text: REGEXMATCH (New)": # This was already there, but keeping it for context
            st.subheader("📝 REGEXMATCH (New)")
            st.info("Creates a new boolean column indicating if the text in a selected column matches a given regular expression pattern.")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                text_col_rm = st.selectbox("Select Text Column", df.columns.tolist(), key="text_rm_col")
                regex_pattern_rm = st.text_input("Regular Expression Pattern:", value="^A.*", help="e.g., '^A.*' for text starting with 'A'", key="text_rm_pattern")
                new_col_name_rm = st.text_input("New Column Name:", value=f"{text_col_rm}_matches_regex", key="text_rm_new_col")

                if st.button("Execute REGEXMATCH", key="excel_execute_regexmatch"):
                    if not regex_pattern_rm.strip() or not new_col_name_rm.strip():
                        st.error("Please provide a regex pattern and a new column name.")
                    else:
                        try:
                            df_copy = df.copy()
                            # Ensure column is string type and handle NaNs
                            text_series = df_copy[text_col_rm].astype(str).fillna('')
                            df_copy[new_col_name_rm] = text_series.str.contains(regex_pattern_rm, regex=True, na=False)
                            
                            st.success(f"Applied REGEXMATCH to '{text_col_rm}'. Created '{new_col_name_rm}'.")
                            st.dataframe(df_copy[[text_col_rm, new_col_name_rm]].head())
                            st.session_state.df = df_copy # Update main DataFrame
                        except Exception as e:
                            st.error(f"Error applying REGEXMATCH: {str(e)}")

        elif operation == "Text: REGEXREPLACE (New)": # This was already there, but keeping it for context
            st.subheader("📝 REGEXREPLACE (New)")
            st.info("Replaces substrings in a selected column that match a regular expression pattern with a specified replacement string.")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                text_col_rr = st.selectbox("Select Text Column", df.columns.tolist(), key="text_rr_col")
                regex_pattern_rr = st.text_input("Regular Expression Pattern to Find:", value="[0-9]+", help="e.g., '[0-9]+' to find numbers", key="text_rr_pattern")
                replacement_str_rr = st.text_input("Replacement String:", value="", help="e.g., '' to remove matches", key="text_rr_replacement")
                new_col_name_rr = st.text_input("New Column Name:", value=f"{text_col_rr}_regex_replaced", key="text_rr_new_col")

                if st.button("Execute REGEXREPLACE", key="excel_execute_regexreplace"):
                    if not regex_pattern_rr.strip() or not new_col_name_rr.strip():
                        st.error("Please provide a regex pattern and a new column name.")
                    else:
                        try:
                            df_copy = df.copy()
                            # Ensure column is string type and handle NaNs
                            text_series = df_copy[text_col_rr].astype(str).fillna('')
                            df_copy[new_col_name_rr] = text_series.str.replace(regex_pattern_rr, replacement_str_rr, regex=True)
                            
                            st.success(f"Applied REGEXREPLACE to '{text_col_rr}'. Created '{new_col_name_rr}'.")
                            st.dataframe(df_copy[[text_col_rr, new_col_name_rr]].head())
                            st.session_state.df = df_copy # Update main DataFrame
                        except Exception as e:
                            st.error(f"Error applying REGEXREPLACE: {str(e)}")

        elif operation == "Text: TEXTJOIN (New)": # This was already there, but keeping it for context
            st.subheader("📝 TEXTJOIN (New)")
            st.info("Combines text from multiple selected columns into a single new column, using a specified delimiter.")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                cols_to_textjoin = st.multiselect("Select Columns to Join", df.columns.tolist(), key="text_tj_cols")
                delimiter_tj = st.text_input("Delimiter:", value=" ", key="text_tj_delimiter")
                ignore_empty_tj = st.checkbox("Ignore Empty Cells", value=True, key="text_tj_ignore_empty")
                new_col_name_tj = st.text_input("New Column Name:", value="Joined_Text", key="text_tj_new_col")

                if st.button("Execute TEXTJOIN", key="excel_execute_textjoin"):
                    if not cols_to_textjoin or not new_col_name_tj.strip():
                        st.error("Please select columns to join and provide a new column name.")
                    else:
                        try:
                            df_copy = df.copy()
                            # Convert selected columns to string, handle NaNs
                            temp_df = df_copy[cols_to_textjoin].astype(str).fillna('')
                            
                            if ignore_empty_tj:
                                # Replace empty strings with NaN so dropna works
                                temp_df = temp_df.replace('', np.nan)
                                df_copy[new_col_name_tj] = temp_df.apply(lambda x: delimiter_tj.join(x.dropna()), axis=1)
                            else:
                                df_copy[new_col_name_tj] = temp_df.apply(lambda x: delimiter_tj.join(x), axis=1)
                            
                            st.success(f"Applied TEXTJOIN to {len(cols_to_textjoin)} columns. Created '{new_col_name_tj}'.")
                            st.dataframe(df_copy[[*cols_to_textjoin, new_col_name_tj]].head())
                            st.session_state.df = df_copy # Update main DataFrame
                        except Exception as e:
                            st.error(f"Error applying TEXTJOIN: {str(e)}")

        elif operation == "Statistical: UNIQUE (New)": # This was already there, but keeping it for context
            st.subheader("📊 UNIQUE (New)")
            st.info("Extracts and displays unique values from selected column(s).")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                cols_for_unique = st.multiselect("Select Column(s) for Unique Values", df.columns.tolist(), key="stat_unique_cols")
                
                if st.button("Get Unique Values", key="excel_get_unique"):
                    if not cols_for_unique:
                        st.warning("Please select at least one column.")
                    else:
                        try:
                            unique_df = df[cols_for_unique].drop_duplicates().reset_index(drop=True)
                            st.success(f"Found {len(unique_df)} unique combination(s).")
                            st.dataframe(unique_df)
                        except Exception as e:
                            st.error(f"Error getting unique values: {str(e)}")

        elif operation == "Math: ROUND/ABS/SQRT": # This was already there, but keeping it for context
            st.subheader("🧮 Math Operations (ROUND, ABS, SQRT)")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                numeric_cols_math1 = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols_math1:
                    st.warning("No numeric columns found for Math operations.")
                else:
                    math_col1 = st.selectbox("Select Numeric Column", numeric_cols_math1, key="math1_col")
                    math_type1 = st.radio("Math Operation", ["ROUND", "ABS", "SQRT"], key="math1_type")
                    
                    if math_type1 == "ROUND":
                        decimals = st.number_input("Number of Decimals", min_value=0, value=2, step=1, key="math1_round_decimals")
                    
                    new_col_name_math1 = st.text_input("New Column Name (optional, modifies in-place if blank):", key="math1_new_col")

                    if st.button("Execute Math Operation", key="excel_execute_math1"):
                        try:
                            df_copy = df.copy()
                            
                            if math_type1 == "ROUND":
                                result_series = df_copy[math_col1].round(decimals)
                            elif math_type1 == "ABS":
                                result_series = df_copy[math_col1].abs()
                            elif math_type1 == "SQRT":
                                # Handle potential negative values for SQRT
                                if (df_copy[math_col1] < 0).any():
                                    st.warning(f"Column '{math_col1}' contains negative values. SQRT will result in NaN for these.")
                                result_series = np.sqrt(df_copy[math_col1])
                            
                            if new_col_name_math1.strip():
                                df_copy[new_col_name_math1] = result_series
                                st.success(f"Applied {math_type1} to '{math_col1}'. Created '{new_col_name_math1}'.")
                                st.dataframe(df_copy[[math_col1, new_col_name_math1]].head())
                                st.session_state.df = df_copy # Update main DataFrame
                            else:
                                # Modify in-place for preview
                                df_copy[math_col1] = result_series
                                st.success(f"Applied {math_type1} to '{math_col1}' (in-place preview).")
                                st.dataframe(df_copy[[math_col1]].head())

                        except Exception as e:
                            st.error(f"Error applying math operation: {str(e)}")

        elif operation == "Math: POWER/MOD": # This was already there, but keeping it for context
            st.subheader("🧮 Math Operations (POWER, MOD)")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                numeric_cols_math2 = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols_math2:
                    st.warning("No numeric columns found for Math operations.")
                else:
                    math_type2 = st.radio("Math Operation", ["POWER", "MOD"], key="math2_type")
                    math_col2 = st.selectbox("Select Numeric Column", numeric_cols_math2, key="math2_col")
                    
                    if math_type2 == "POWER":
                        power_value = st.number_input("Power:", value=2.0, step=0.1, key="math2_power_value")
                        new_col_name_math2 = st.text_input("New Column Name:", value=f"{math_col2}_power_{power_value}", key="math2_new_col")
                    elif math_type2 == "MOD":
                        divisor_value = st.number_input("Divisor:", value=2.0, step=0.1, key="math2_mod_divisor")
                        new_col_name_math2 = st.text_input("New Column Name:", value=f"{math_col2}_mod_{divisor_value}", key="math2_new_col")

                    if st.button("Execute Math Operation", key="excel_execute_math2"):
                         if not new_col_name_math2.strip():
                             st.error("Please provide a name for the new column.")
                         else:
                            try:
                                df_copy = df.copy()
                                operation_successful = False
                                
                                if math_type2 == "POWER":
                                    result_series = df_copy[math_col2] ** power_value
                                    operation_successful = True
                                elif math_type2 == "MOD":
                                    # Handle division by zero
                                    if divisor_value == 0:
                                        st.error("Divisor cannot be zero for MOD operation.")
                                        # operation_successful remains False
                                    else:
                                        result_series = df_copy[math_col2] % divisor_value
                                        operation_successful = True
                                
                                if operation_successful:
                                    df_copy[new_col_name_math2] = result_series
                                    st.success(f"Applied {math_type2} to '{math_col2}'. Created '{new_col_name_math2}'.")
                                    st.dataframe(df_copy[[math_col2, new_col_name_math2]].head())
                                    st.session_state.df = df_copy # Update main DataFrame
                            except Exception as e:
                                st.error(f"Error applying math operation: {str(e)}")

        elif operation == "Statistical: RANK/PERCENTILE": # This was already there, but keeping it for context
            st.subheader("📊 Statistical Operations (RANK, PERCENTILE)")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                numeric_cols_stat = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols_stat:
                    st.warning("No numeric columns found for Statistical operations.")
                else:
                    stat_col = st.selectbox("Select Numeric Column", numeric_cols_stat, key="stat_col")
                    stat_type = st.radio("Statistical Operation", ["RANK", "PERCENTILE"], key="stat_type")
                    
                    if stat_type == "RANK":
                        rank_method = st.selectbox("Rank Method", ["average", "min", "max", "first", "dense"], key="stat_rank_method")
                        rank_ascending = st.checkbox("Ascending Rank", value=True, key="stat_rank_ascending")
                        new_col_name_rank = st.text_input("New Column Name:", value=f"{stat_col}_rank", key="stat_rank_new_col")
                    elif stat_type == "PERCENTILE":
                        percentile_value = st.slider("Percentile (0-100)", 0, 100, 50, key="stat_percentile_value")
                        new_col_name_percentile = st.text_input("New Column Name:", value=f"{stat_col}_percentile", key="stat_percentile_new_col")

                    if st.button("Execute Statistical Operation", key="excel_execute_stat"):
                         if (stat_type == "RANK" and not new_col_name_rank.strip()) or (stat_type == "PERCENTILE" and not new_col_name_percentile.strip()):
                             st.error("Please provide a name for the new column.")
                         else:
                            try:
                                df_copy = df.copy()
                                
                                if stat_type == "RANK":
                                    df_copy[new_col_name_rank] = df_copy[stat_col].rank(method=rank_method, ascending=rank_ascending)
                                    st.success(f"Applied RANK to '{stat_col}'. Created '{new_col_name_rank}'.")
                                    st.dataframe(df_copy[[stat_col, new_col_name_rank]].head())
                                elif stat_type == "PERCENTILE":
                                    # Calculate the percentile value across the series
                                    percentile_val = df_copy[stat_col].quantile(percentile_value / 100.0)
                                    # Create a new column indicating if the value is >= the percentile value (or just store the percentile value itself?)
                                    # Excel's PERCENTILE.INC/EXC returns a single value. Let's return the value.
                                    st.metric(f"{percentile_value}th Percentile of '{stat_col}'", f"{percentile_val:.2f}")
                                    # Or, create a column indicating if each row is above/below that percentile? Let's do the latter for a new column.
                                    df_copy[new_col_name_percentile] = (df_copy[stat_col] >= percentile_val).astype(int) # 1 if >= percentile, 0 otherwise
                                    st.success(f"Calculated {percentile_value}th percentile ({percentile_val:.2f}) for '{stat_col}'. Created '{new_col_name_percentile}' (1 if >= percentile, 0 otherwise).")
                                    st.dataframe(df_copy[[stat_col, new_col_name_percentile]].head())

                                st.session_state.df = df_copy # Update main DataFrame

                            except Exception as e:
                                st.error(f"Error applying statistical operation: {str(e)}")

        elif operation == "Logical: IF (Conditional Column)": # This was already there, but keeping it for context
            st.subheader("🧠 Logical IF (Create Conditional Column)")
            st.info("Creates a new column based on a condition applied to another column.")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                condition_col_if = st.selectbox("Select Column for Condition", df.columns.tolist(), key="if_condition_col")
                condition_logic = st.text_input("Condition Logic (e.g., > 100, == 'Active', .isnull()):", key="if_condition_logic")
                value_if_true = st.text_input("Value if True:", key="if_value_true")
                value_if_false = st.text_input("Value if False:", key="if_value_false")
                new_col_name_if = st.text_input("New Column Name:", value=f"{condition_col_if}_conditional", key="if_new_col")

                if st.button("Execute IF", key="excel_execute_if"):
                    if not condition_logic.strip() or not new_col_name_if.strip():
                        st.error("Please provide condition logic and a new column name.")
                    else:
                        try:
                            df_copy = df.copy()
                            
                            # Safely evaluate the condition logic
                            # This is still using eval, caution advised.
                            # A safer approach would be specific UI for common conditions (>, <, ==, is null, contains)
                            condition_expr = f"df_copy['{condition_col_if}']{condition_logic}"
                            condition_mask = eval(condition_expr)
                            
                            df_copy[new_col_name_if] = np.where(condition_mask, value_if_true, value_if_false)
                            
                            st.success(f"Applied IF logic to '{condition_col_if}'. Created '{new_col_name_if}'.")
                            st.dataframe(df_copy[[condition_col_if, new_col_name_if]].head())
                            st.session_state.df = df_copy # Update main DataFrame

                        except Exception as e:
                            st.error(f"Error applying IF logic: {str(e)}")
                            st.info("Ensure your condition logic is valid Python syntax for a boolean expression on a Pandas Series.")

        elif operation == "Data: Transpose": # This was already there, but keeping it for context
            st.subheader("↔️ Transpose DataFrame")
            st.info("Swaps rows and columns. The original index becomes columns, and original columns become the index.")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                if st.button("Execute Transpose", key="excel_execute_transpose"):
                    try:
                        transposed_df = df.T
                        st.success(f"DataFrame transposed. New shape: {transposed_df.shape}")
                        st.dataframe(transposed_df)
                        st.session_state.df_transposed_temp = transposed_df # Store for potential download
                    except Exception as e:
                        st.error(f"Error transposing DataFrame: {str(e)}")

        elif operation == "Data: Fill Down/Up": # This was already there, but keeping it for context
            st.subheader("💧 Fill Missing Values (Fill Down/Up)")
            st.info("Fills NaN values using the previous (Fill Down) or next (Fill Up) valid observation.")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                fill_col_fillna = st.selectbox("Select Column to Fill", df.columns.tolist(), key="fillna_col")
                fill_direction = st.radio("Fill Direction", ["Fill Down (ffill)", "Fill Up (bfill)"], key="fillna_direction")
                
                if st.button("Execute Fill", key="excel_execute_fillna"):
                    try:
                        df_copy = df.copy()
                        if fill_direction == "Fill Down (ffill)":
                            df_copy[fill_col_fillna].fillna(method='ffill', inplace=True)
                        elif fill_direction == "Fill Up (bfill)":
                            df_copy[fill_col_fillna].fillna(method='bfill', inplace=True)
                        
                        st.success(f"Applied {fill_direction} to '{fill_col_fillna}'.")
                        st.dataframe(df_copy[[fill_col_fillna]].head())
                        st.session_state.df = df_copy # Update main DataFrame

                    except Exception as e:
                        st.error(f"Error applying fill operation: {str(e)}")

        elif operation == "Date/Time: Extract Component": # This was already there, but keeping it for context
            st.subheader("⏰ Extract Date/Time Component")
            st.info("Extracts components like Year, Month, Day, Hour, etc., from a datetime column.")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                datetime_cols_extract = df.select_dtypes(include=['datetime64']).columns.tolist()
                if not datetime_cols_extract:
                    st.warning("No datetime columns found. Please convert a column to datetime first (e.g., in Data Upload or EDA).")
                else:
                    col_to_extract_dt = st.selectbox("Select Datetime Column", datetime_cols_extract, key="dt_extract_col")
                    component_to_extract = st.selectbox("Select Component to Extract", ["year", "month", "day", "hour", "minute", "second", "dayofweek", "dayofyear", "week", "quarter"], key="dt_extract_component")
                    new_col_name_dt_extract = st.text_input("New Column Name:", value=f"{col_to_extract_dt}_{component_to_extract}", key="dt_extract_new_col")

                    if st.button("Execute Extraction", key="excel_execute_dt_extract"):
                         if not new_col_name_dt_extract.strip():
                             st.error("Please provide a name for the new column.")
                         else:
                            try:
                                df_copy = df.copy()
                                # Use .dt accessor
                                df_copy[new_col_name_dt_extract] = getattr(df_copy[col_to_extract_dt].dt, component_to_extract)
                                
                                st.success(f"Extracted '{component_to_extract}' from '{col_to_extract_dt}'. Created '{new_col_name_dt_extract}'.")
                                st.dataframe(df_copy[[col_to_extract_dt, new_col_name_dt_extract]].head())
                                st.session_state.df = df_copy # Update main DataFrame

                            except Exception as e:
                                st.error(f"Error extracting datetime component: {str(e)}")

        elif operation == "Goal Seek (New)": # This was already there, but keeping it for context
            st.subheader("🎯 Goal Seek")
            st.info("Find an input value (`x`) for a formula to achieve a desired target value. This uses an iterative solver.")
            st.warning("⚠️ This tool uses `eval()` to process the formula, which can be a security risk. Use with trusted data and formulas only.")
            
            try:
                from scipy.optimize import root_scalar
            except ImportError:
                st.error("Scipy is required for Goal Seek. Please install it: `pip install scipy`")
                st.stop()

            st.markdown("#### 1. Define the Formula")
            formula_str = st.text_input(
                "Formula (must include 'x' as the variable to change)", 
                value="(df['price'] * (1 - x)) - df['cost']",
                help="Use 'df' for the DataFrame and 'x' for the variable you want to solve for. E.g., to find a discount `x` that affects profit."
            )

            st.markdown("#### 2. Set the Target")
            c1, c2 = st.columns(2)
            with c1:
                agg_func_gs = st.selectbox("Aggregate the formula result using:", ["sum", "mean"], key="gs_agg_func")
            with c2:
                target_value_gs = st.number_input("Set aggregated result to value:", value=100000.0, format="%.2f", key="gs_target")

            st.markdown("#### 3. Configure the Solver")
            c3, c4 = st.columns(2)
            with c3:
                bracket_low = st.number_input("Search Bracket (Low)", value=-1.0, key="gs_bracket_low")
            with c4:
                bracket_high = st.number_input("Search Bracket (High)", value=1.0, key="gs_bracket_high")

            if st.button("Run Goal Seek", key="excel_execute_gs"):
                if 'x' not in formula_str:
                    st.error("The formula must contain 'x' as the variable to be changed.")
                else:
                    with st.spinner("Seeking solution..."):
                        try:
                            # Define the objective function for the solver
                            def objective_func(x, df_obj, formula, agg_func, target):
                                # This is where the user's formula is evaluated
                                result_series = eval(formula, {'df': df_obj, 'np': np, 'x': x})
                                current_agg = result_series.agg(agg_func)
                                return current_agg - target

                            # Run the solver
                            sol = root_scalar(
                                objective_func, 
                                args=(df, formula_str, agg_func_gs, target_value_gs),
                                bracket=[bracket_low, bracket_high],
                                method='brentq'
                            )

                            if sol.converged:
                                st.success(f"Solution Found!")
                                st.metric("Required value for `x`", f"{sol.root:.6f}")
                                st.write(f"The solver converged in {sol.function_calls} iterations.")
                            else:
                                st.error("Solver did not converge. Try adjusting the search bracket or checking your formula.")
                                st.write(sol)

                        except Exception as e:
                            st.error(f"An error occurred during Goal Seek: {e}")
                            st.info("Check if your formula is valid and the columns exist.")

        elif operation == "Conditional Formatting (New)": # This was already there, but keeping it for context
            st.subheader("🎨 Conditional Formatting")
            st.info("Define rules to visually style your data. The formatting is applied to the displayed table below.")

            if 'cf_rules' not in st.session_state:
                st.session_state.cf_rules = [{'Column': df.columns[0], 'Condition': 'Greater Than', 'Value': '100', 'Style': 'background-color: lightgreen'}]

            edited_rules_cf = st.data_editor(
                st.session_state.cf_rules,
                num_rows="dynamic",
                column_config={
                    "Column": st.column_config.SelectboxColumn("Column", options=df.columns.tolist(), required=True),
                    "Condition": st.column_config.SelectboxColumn("Condition", options=["Greater Than", "Less Than", "Equals", "Contains"], required=True),
                    "Value": st.column_config.TextColumn("Value", required=True),
                    "Style": st.column_config.SelectboxColumn("Style", options=['background-color: lightgreen', 'background-color: lightcoral', 'background-color: lightyellow', 'font-weight: bold; color: blue'], required=True)
                },
                key="excel_cf_rules_editor"
            )

            def apply_styles(val, rules, col_name):
                style = ''
                for rule in rules:
                    if rule['Column'] == col_name:
                        # Simplified logic for demo
                        if rule['Condition'] == 'Greater Than' and pd.to_numeric(val, errors='coerce') > pd.to_numeric(rule['Value'], errors='coerce'):
                            style = rule['Style']
                return style

            st.dataframe(df.head(100).style.apply(lambda x: [f'background-color: {"lightcoral" if v > 0 else "lightgreen"}' if (isinstance(v, (int,float)) and x.name in [r['Column'] for r in edited_rules_cf if r['Condition']=='Greater Than' and v > pd.to_numeric(r['Value'], errors='coerce')]) else '' for v in x], axis=0))
            st.warning("Conditional formatting preview is limited for performance. Full styling would be applied on export.")

elif selected_tool == "💼 Power BI Style Dashboard": # This was already there, but keeping it for context
    st.markdown('<h2 class="tool-header">💼 Power BI Style Dashboard</h2>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload data first to build a dashboard!")
    else:
        df = st.session_state.df
        
        # --- Sidebar for Filters ---
        st.sidebar.header("Dashboard Filters")
        
        # Use a copy of the dataframe for filtering
        filtered_df = df.copy()
        
        # Get column types
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Create filters for categorical columns
        for col in categorical_cols:
            unique_vals = ['All'] + sorted(df[col].unique().tolist())
            selected_vals = st.sidebar.multiselect(f"Filter by {col}", unique_vals, default='All')
            if 'All' not in selected_vals and selected_vals:
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

        # Create filters for numeric columns (range sliders)
        for col in numeric_cols:
            min_val, max_val = float(df[col].min()), float(df[col].max())
            if min_val < max_val:
                selected_range = st.sidebar.slider(f"Filter by {col} range", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[(filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])]

        # --- Main Dashboard Area ---
        st.subheader("📊 Key Performance Indicators (KPIs)")
        
        # Configure KPIs
        kpi_cols = st.multiselect("Select up to 5 columns for KPIs", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
        
        if kpi_cols:
            kpi_display_cols = st.columns(len(kpi_cols))
            for i, col in enumerate(kpi_cols):
                with kpi_display_cols[i]:
                    # Calculate metric on the filtered dataframe
                    total = filtered_df[col].sum()
                    average = filtered_df[col].mean()
                    count = filtered_df[col].count()
                    st.metric(label=f"Total {col}", value=f"{total:,.2f}")
                    st.metric(label=f"Average {col}", value=f"{average:,.2f}")
                    st.metric(label=f"Count of {col}", value=f"{count:,}")
        else:
            st.info("Select numeric columns from the dropdown above to display KPIs.")

        st.markdown("---")
        st.subheader("📈 Dashboard Charts")
        
        # Helper function to create a single chart, now self-contained for column lists
        def create_dashboard_chart(chart_num, filtered_df):
            st.markdown(f"#### Chart {chart_num}")

            # Ensure there's data to plot
            if filtered_df.empty:
                st.info(f"Chart {chart_num}: Filtered data is empty.")
                return

            # Helper to safely get a column from a list, cycling through
            def get_cycled_col(col_list, index_offset):
                if col_list:
                    return col_list[index_offset % len(col_list)]
                return None

            # Helper for automatic chart parameter selection
            def _auto_select_chart_params(chart_num, numeric_cols, categorical_cols, all_cols_df):
                chart_types_available = [
                    "Bar", "Pie", "Line", "Scatter", "Histogram", "Box", "Violin", "Area",
                    "Density Heatmap", "Treemap", "Sunburst", "Funnel", "Polar Bar", "Polar Scatter"
                ]
                chart_type = chart_types_available[(chart_num - 1) % len(chart_types_available)]

                x_col, y_col, names_col, values_col, color_col, path_cols, r_col, theta_col = None, None, None, None, None, None, None, None

                # Select color column
                if len(categorical_cols) > 0:
                    color_col = get_cycled_col(categorical_cols, chart_num)

                # Logic for each chart type
                if chart_type == "Bar":
                    if len(categorical_cols) > 0:
                        x_col = get_cycled_col(categorical_cols, chart_num - 1)
                        if len(numeric_cols) > 0:
                            y_col = get_cycled_col(numeric_cols, chart_num - 1)
                elif chart_type == "Pie":
                    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        names_col = get_cycled_col(categorical_cols, chart_num - 1)
                        values_col = get_cycled_col(numeric_cols, chart_num - 1)
                elif chart_type in ["Line", "Area"]:
                    datetime_cols = filtered_df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
                    if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                        x_col = get_cycled_col(datetime_cols, 0)
                        y_col = get_cycled_col(numeric_cols, chart_num - 1)
                    elif len(numeric_cols) > 1:
                        x_col = get_cycled_col(numeric_cols, chart_num - 1)
                        y_col = get_cycled_col(numeric_cols, chart_num)
                elif chart_type in ["Scatter", "Density Heatmap", "Density Contour"]:
                    if len(numeric_cols) > 1:
                        x_col = get_cycled_col(numeric_cols, chart_num - 1)
                        y_col = get_cycled_col(numeric_cols, chart_num)
                elif chart_type == "Histogram":
                    if len(numeric_cols) > 0:
                        x_col = get_cycled_col(numeric_cols, chart_num - 1)
                elif chart_type in ["Box", "Violin"]:
                    if len(numeric_cols) > 0:
                        y_col = get_cycled_col(numeric_cols, chart_num - 1)
                        if len(categorical_cols) > 0:
                            x_col = get_cycled_col(categorical_cols, chart_num - 1)
                elif chart_type in ["Treemap", "Sunburst"]:
                    if len(categorical_cols) > 1 and len(numeric_cols) > 0:
                        path_cols = [get_cycled_col(categorical_cols, i) for i in range(min(2, len(categorical_cols)))]
                        values_col = get_cycled_col(numeric_cols, chart_num - 1)
                elif chart_type == "Funnel":
                     if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        y_col = get_cycled_col(categorical_cols, chart_num - 1)
                        x_col = get_cycled_col(numeric_cols, chart_num - 1)
                elif chart_type in ["Polar Bar", "Polar Scatter"]:
                    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        theta_col = get_cycled_col(categorical_cols, chart_num - 1)
                        r_col = get_cycled_col(numeric_cols, chart_num - 1)
                
                if not any([x_col, y_col, names_col, path_cols, r_col, theta_col]):
                    if len(numeric_cols) > 0:
                        chart_type, x_col = "Histogram", get_cycled_col(numeric_cols, 0)
                    elif len(categorical_cols) > 0:
                        chart_type, x_col = "Bar", get_cycled_col(categorical_cols, 0)
                    else:
                        chart_type = None

                return chart_type, x_col, y_col, names_col, values_col, color_col, path_cols, r_col, theta_col

            # Dynamically get numeric and categorical columns from the filtered_df
            current_numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
            current_categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
            all_cols = filtered_df.columns.tolist()

            # Initialize plot parameters
            chart_type, x_col, y_col, names_col, values_col, color_col, path_cols, r_col, theta_col = [None] * 9

            # --- Manual Chart Configuration for specific charts ---
            if chart_num in [1, 5, 9]:
                st.markdown(f"##### Manual Configuration for Chart {chart_num}")
                chart_types_manual = [
                    "Auto", "Bar", "Line", "Scatter", "Pie", "Histogram", "Box", "Violin", "Area",
                    "Density Heatmap", "Density Contour", "Treemap", "Sunburst", "Funnel", "Polar Bar", "Polar Scatter"
                ]
                manual_chart_type = st.selectbox(
                    "Select Chart Type",
                    chart_types_manual,
                    key=f"manual_chart_type_{chart_num}"
                )

                if manual_chart_type == "Auto":
                    chart_type, x_col, y_col, names_col, values_col, color_col, path_cols, r_col, theta_col = _auto_select_chart_params(
                        chart_num, current_numeric_cols, current_categorical_cols, all_cols
                    )
                else:
                    chart_type = manual_chart_type
                    
                    # Generic color selector
                    color_col = st.selectbox(f"Color by (Optional, Chart {chart_num})", ['None'] + all_cols, key=f"manual_color_col_{chart_num}")
                    if color_col == 'None': color_col = None

                    if chart_type in ["Bar", "Line", "Scatter", "Area", "Histogram", "Box", "Violin", "Density Heatmap", "Density Contour", "Funnel"]:
                        x_col = st.selectbox(f"X-axis (Chart {chart_num})", ['None'] + all_cols, key=f"manual_x_col_{chart_num}")
                        if x_col == 'None': x_col = None
                        if chart_type not in ["Histogram"]:
                            y_col = st.selectbox(f"Y-axis (Chart {chart_num})", ['None'] + all_cols, key=f"manual_y_col_{chart_num}")
                            if y_col == 'None': y_col = None
                    
                    elif chart_type == "Pie":
                        names_col = st.selectbox(f"Names (Categorical, Chart {chart_num})", ['None'] + current_categorical_cols, key=f"manual_names_col_{chart_num}")
                        if names_col == 'None': names_col = None
                        values_col = st.selectbox(f"Values (Numeric, Chart {chart_num})", ['None'] + current_numeric_cols, key=f"manual_values_col_{chart_num}")
                        if values_col == 'None': values_col = None

                    elif chart_type in ["Treemap", "Sunburst"]:
                        path_cols = st.multiselect(f"Path/Hierarchy (Categorical, Chart {chart_num})", current_categorical_cols, key=f"manual_path_cols_{chart_num}")
                        values_col = st.selectbox(f"Values (Numeric, Chart {chart_num})", ['None'] + current_numeric_cols, key=f"manual_values_col_{chart_num}")
                        if values_col == 'None': values_col = None
                    
                    elif chart_type in ["Polar Bar", "Polar Scatter"]:
                        r_col = st.selectbox(f"R (radius, Numeric, Chart {chart_num})", ['None'] + current_numeric_cols, key=f"manual_r_col_{chart_num}")
                        if r_col == 'None': r_col = None
                        theta_col = st.selectbox(f"Theta (angle, Categorical, Chart {chart_num})", ['None'] + current_categorical_cols, key=f"manual_theta_col_{chart_num}")
                        if theta_col == 'None': theta_col = None
            else:
                # --- Existing Automatic Chart Selection Logic ---
                chart_type, x_col, y_col, names_col, values_col, color_col, path_cols, r_col, theta_col = _auto_select_chart_params(
                    chart_num, current_numeric_cols, current_categorical_cols, all_cols
                )

            if chart_type is None:
                st.info(f"Chart {chart_num}: Not enough suitable columns to generate a plot.")
                return

            # Add a check for color_col validity before plotting
            if color_col:
                if color_col not in filtered_df.columns or filtered_df[color_col].nunique(dropna=True) <= 1:
                    st.info(f"Chart {chart_num}: Color column '{color_col}' not suitable. Disabling color.")
                    color_col = None

            try:
                fig = None
                title = f"Chart {chart_num}: {chart_type}"
                
                if chart_type == "Bar":
                    if x_col:
                        if y_col and y_col in current_numeric_cols:
                            group_cols = [x_col]
                            if color_col and color_col != x_col and color_col in current_categorical_cols:
                                group_cols.append(color_col)
                            plot_df = filtered_df.groupby(group_cols, as_index=False)[y_col].sum()
                            # Validate color_col against plot_df
                            effective_color_col = color_col if color_col in plot_df.columns and color_col != x_col else None
                            fig = px.bar(plot_df, x=x_col, y=y_col, color=effective_color_col, title=f"{y_col} by {x_col}")
                        else:
                            value_counts = filtered_df[x_col].value_counts().reset_index()
                            value_counts.columns = [x_col, 'Count']
                            # For value_counts, color can only be x_col itself
                            effective_color_col = x_col if color_col == x_col else None
                            fig = px.bar(value_counts, x=x_col, y='Count', color=effective_color_col, title=f"Count of {x_col}")
                
                elif chart_type == "Line":
                    if x_col and y_col and y_col in current_numeric_cols:
                        fig = px.line(filtered_df.sort_values(by=x_col), x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")

                elif chart_type == "Scatter":
                    if x_col and y_col and x_col in current_numeric_cols and y_col in current_numeric_cols:
                        fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")

                elif chart_type == "Pie":
                    if names_col and values_col and values_col in current_numeric_cols:
                        fig = px.pie(filtered_df, names=names_col, values=values_col, title=f"Distribution of {values_col} by {names_col}")
                
                elif chart_type == "Histogram":
                    if x_col and x_col in current_numeric_cols:
                        fig = px.histogram(filtered_df, x=x_col, color=color_col, title=f"Histogram of {x_col}")

                elif chart_type in ["Box", "Violin"]:
                    if y_col and y_col in current_numeric_cols:
                        plot_func = px.box if chart_type == "Box" else px.violin
                        fig = plot_func(filtered_df, y=y_col, x=x_col, color=color_col, title=f"{chart_type} Plot of {y_col}")

                elif chart_type == "Area":
                    if x_col and y_col and y_col in current_numeric_cols:
                        fig = px.area(filtered_df.sort_values(by=x_col), x=x_col, y=y_col, color=color_col, title=f"Area Plot of {y_col} over {x_col}")

                elif chart_type in ["Density Heatmap", "Density Contour"]:
                    if x_col and y_col and x_col in current_numeric_cols and y_col in current_numeric_cols:
                        plot_func = px.density_heatmap if chart_type == "Density Heatmap" else px.density_contour
                        fig = plot_func(filtered_df, x=x_col, y=y_col, title=f"{chart_type} of {y_col} vs {x_col}")

                elif chart_type in ["Treemap", "Sunburst"]:
                    if path_cols and values_col and values_col in current_numeric_cols:
                        plot_func = px.treemap if chart_type == "Treemap" else px.sunburst
                        fig = plot_func(filtered_df, path=path_cols, values=values_col, color=color_col, title=f"{chart_type} of {values_col}")

                elif chart_type == "Funnel":
                    if x_col and y_col and x_col in current_numeric_cols and y_col in current_categorical_cols:
                        plot_df = filtered_df.groupby(y_col, as_index=False)[x_col].sum()
                        fig = px.funnel(plot_df, x=x_col, y=y_col, title=f"Funnel Chart of {x_col} by {y_col}")

                elif chart_type in ["Polar Bar", "Polar Scatter"]:
                    if r_col and theta_col and r_col in current_numeric_cols and theta_col in current_categorical_cols:
                        plot_func = px.bar_polar if chart_type == "Polar Bar" else px.scatter_polar
                        if chart_type == "Polar Bar":
                            plot_df = filtered_df.groupby(theta_col, as_index=False)[r_col].mean()
                            # Validate color_col against plot_df
                            effective_color_col = color_col if color_col in plot_df.columns and color_col != theta_col else None
                            fig = plot_func(plot_df, r=r_col, theta=theta_col, color=effective_color_col, title=f"Polar Bar Chart")
                        else: # Polar Scatter
                            fig = plot_func(filtered_df, r=r_col, theta=theta_col, color=color_col, title=f"Polar Scatter Chart")

                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_plot_{chart_num}_{chart_type}_{x_col}_{y_col}_{names_col}_{values_col}_{color_col}")
                else:
                    st.info(f"Chart {chart_num}: Could not generate a '{chart_type}' plot with the selected columns or available data.")

            except Exception as e:
                st.error(f"Error generating Chart {chart_num} ({chart_type}) with columns (x:{x_col}, y:{y_col}, names:{names_col}, values:{values_col}, path:{path_cols}, r:{r_col}, theta:{theta_col}): {e}")

        # Allow user to select the number of charts to display
        num_charts_to_display = st.slider(
            "Number of Charts to Display",
            min_value=1,
            max_value=20, # Set a reasonable maximum number of charts
            value=10, # Default to 10 charts
            step=1,
            help="Adjust the number of charts shown on the dashboard. More charts may impact performance."
        )

        # Loop to create charts based on user selection
        for i in range(1, num_charts_to_display + 1):
            if i % 2 != 0: # Start a new row for odd numbered charts
                chart_layout_cols = st.columns(2)
            
            with chart_layout_cols[(i - 1) % 2]: # Place in first or second column
                create_dashboard_chart(i, filtered_df)

elif selected_tool == "🐼 Pandas Query Tool": # This was already there, but keeping it for context
    st.markdown('<h2 class="tool-header">🐼 Advanced Pandas Query Tool</h2>', unsafe_allow_html=True)

    # --- Pandas Query Examples ---
    with st.expander("🐼 Pandas Query Examples", expanded=False):
        st.markdown("""
**1. Load the data (Illustrative - data is already loaded as `df` in the app)**
```python
import pandas as pd
df = pd.read_csv("your_file.csv")
```
**2. Show first 5 rows**
```python
df.head()
```
**3. Filter: Only Male users in Tokyo**
```python
df[(df['gender'] == 'Male') & (df['city'] == 'Tokyo')]
```
**4. Count how many purchases per city**
```python
df.groupby('city')['purchases'].sum()
```
**5. Average income per gender**
```python
df.groupby('gender')['income'].mean()
```
**6. Fill missing ages with average**
```python
df['age'] = df['age'].fillna(df['age'].mean())
```
        """)

    if st.session_state.df is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.df
        
        st.subheader("🔧 Pandas Operations")
        
        # Quick operations
        quick_ops = st.selectbox("Quick Operations", [
            "Data Info", "Head/Tail", "Describe", "Value Counts", "Group By (Enhanced)", 
            "Merge/Join", "Pivot Table", "Window Functions", "Melt", "Apply Function", "Query", "Custom"
        ])
        
        if quick_ops == "Data Info":
            col1, col2 = st.columns(2)
            with col1:
                st.code(f"""df.info()
Shape: {df.shape}
Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB""")
            with col2:
                st.dataframe(pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null': df.count(),
                    'Null': df.isnull().sum()
                }))
        
        elif quick_ops == "Head/Tail":
            col1, col2 = st.columns(2)
            n_rows = st.slider("Number of rows", 1, 20, 5)
            
            with col1:
                st.subheader("🔝 Head")
                st.dataframe(df.head(n_rows))
            with col2:
                st.subheader("🔚 Tail")
                st.dataframe(df.tail(n_rows))
        
        elif quick_ops == "Describe":
            describe_type = st.radio("Description Type", ["Numeric Only", "All Columns", "Object Only"])
            
            if describe_type == "Numeric Only":
                st.dataframe(df.describe())
            elif describe_type == "All Columns":
                st.dataframe(df.describe(include='all'))
            else:
                st.dataframe(df.describe(include=['object']))
        
        elif quick_ops == "Value Counts":
            col = st.selectbox("Select Column", df.columns.tolist())
            n_top = st.slider("Top N values", 5, 50, 10)
            
            value_counts = df[col].value_counts().head(n_top)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(value_counts)
            with col2:
                fig = px.bar(x=value_counts.index.astype(str), y=value_counts.values.astype(float), 
                           title=f"Top {n_top} Values in {col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif quick_ops == "Group By (Enhanced)":
            st.subheader("📊 Group By & Aggregate (Enhanced)")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                group_by_cols = st.multiselect(
                    "Select Column(s) to Group By", 
                    df.columns.tolist(), 
                    default=df.columns[0] if len(df.columns) > 0 else None,
                    key="pd_groupby_group_cols"
                )
                
                st.markdown("##### Define Aggregations:")
                
                # Initialize aggregations in session state if not present
                if 'pd_aggregations' not in st.session_state:
                    st.session_state.pd_aggregations = [{"Source Column": df.columns[0] if len(df.columns) > 0 else "", 
                                                         "Aggregation Function": "sum", 
                                                         "Output Column Name": "sum_of_col"}]

                # Use st.data_editor for dynamic aggregation rules
                edited_aggregations = st.data_editor(
                    st.session_state.pd_aggregations,
                    num_rows="dynamic",
                    column_config={
                        "Source Column": st.column_config.SelectboxColumn("Source Column", help="Column to aggregate", options=df.columns.tolist(), required=True),
                        "Aggregation Function": st.column_config.SelectboxColumn("Function", help="Aggregation function", options=["sum", "mean", "count", "min", "max", "std", "nunique", "first", "last"], required=True),
                        "Output Column Name": st.column_config.TextColumn("Output Name", help="Name for the aggregated column", required=True)
                    },
                    key="pd_groupby_data_editor"
                )
                st.session_state.pd_aggregations = edited_aggregations # Update session state

                if st.button("Execute Enhanced Group By", key="pd_execute_groupby"):
                    if not group_by_cols:
                        st.error("Please select at least one column to group by.")
                    elif not edited_aggregations or all(not row["Source Column"] or not row["Output Column Name"] for row in edited_aggregations):
                        st.error("Please define at least one valid aggregation rule.")
                    else:
                        try:
                            agg_dict = {
                                row["Output Column Name"]: (row["Source Column"], row["Aggregation Function"])
                                for row in edited_aggregations if row["Source Column"] and row["Output Column Name"]
                            }
                            if not agg_dict:
                                st.error("No valid aggregations defined.")
                            else:
                                result = df.groupby(group_by_cols, as_index=False).agg(**agg_dict)
                                st.dataframe(result)
                                # Simple visualization: bar chart of the first group_by col vs first aggregated col
                                if not result.empty and len(group_by_cols) > 0 and len(agg_dict) > 0:
                                    first_agg_output_name = list(agg_dict.keys())[0]
                                    fig = px.bar(result, x=group_by_cols[0], y=first_agg_output_name,
                                               title=f"Grouped Data: {group_by_cols[0]} vs {first_agg_output_name}")
                                    st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error executing Group By: {str(e)}")

        elif quick_ops == "Merge/Join":
            st.subheader("🔗 Merge/Join DataFrames")
            st.info("This example demonstrates merging the current DataFrame with itself. Adapt for merging two distinct DataFrames.")

            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    how = st.selectbox("Merge Type (how)", ["left", "right", "outer", "inner", "cross"], key="pd_merge_how")
                    left_on_cols = st.multiselect("Left On Columns (Key(s) from current DataFrame)", df.columns.tolist(), key="pd_merge_left_on")
                    use_index_left = st.checkbox("Use Index for Left Join Key", key="pd_merge_left_index")
                
                with col2:
                    # For self-merge, right_on_cols will also be from df.columns
                    right_on_cols = st.multiselect("Right On Columns (Key(s) from current DataFrame for self-join)", df.columns.tolist(), key="pd_merge_right_on")
                    use_index_right = st.checkbox("Use Index for Right Join Key", key="pd_merge_right_index")

                suffixes_left = st.text_input("Suffix for Left Overlapping Columns", value="_left", key="pd_merge_suffix_left")
                suffixes_right = st.text_input("Suffix for Right Overlapping Columns", value="_right", key="pd_merge_suffix_right")

                if st.button("Execute Merge/Join", key="pd_execute_merge"):
                    # Validate keys if not using index
                    if not use_index_left and not left_on_cols:
                        st.error("Please select 'Left On Columns' or check 'Use Index for Left Join Key'.")
                    elif not use_index_right and not right_on_cols:
                         st.error("Please select 'Right On Columns' or check 'Use Index for Right Join Key'.")
                    elif how == "cross" and (left_on_cols or right_on_cols or use_index_left or use_index_right):
                        st.warning("For 'cross' merge, join keys ('on' columns or index) are not used. They will be ignored.")
                        # Proceed with cross merge logic, potentially clearing keys
                        left_on_cols, right_on_cols, use_index_left, use_index_right = None, None, False, False
                        try:
                            merged_df = pd.merge(
                                df, df.copy(), # Using df.copy() for the right side
                                how=how,
                                suffixes=(suffixes_left, suffixes_right)
                            )
                            st.success(f"Merge completed. Resulting DataFrame has {len(merged_df)} rows and {len(merged_df.columns)} columns.")
                            st.dataframe(merged_df.head())
                            st.session_state.df_merged_temp = merged_df # Store for potential further use or download
                        except Exception as e:
                            st.error(f"Error during merge: {str(e)}")
                    else:
                        # Execute merge for non-cross types
                        try:
                            merged_df = pd.merge(
                                df, df.copy(), # Using df.copy() for the right side
                                how=how,
                                left_on=left_on_cols if not use_index_left else None,
                                right_on=right_on_cols if not use_index_right else None,
                                left_index=use_index_left,
                                right_index=use_index_right,
                                suffixes=(suffixes_left, suffixes_right)
                            )
                            st.success(f"Merge completed. Resulting DataFrame has {len(merged_df)} rows and {len(merged_df.columns)} columns.")
                            st.dataframe(merged_df.head())
                            st.session_state.df_merged_temp = merged_df # Store for potential further use or download
                        except Exception as e:
                            st.error(f"Error during merge: {str(e)}")

        elif quick_ops == "Pivot Table":
            st.subheader("📊 Pivot Table")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                index_cols = st.multiselect("Index (Rows)", df.columns.tolist(), key="pd_pivot_index")
                columns_cols = st.multiselect("Columns", df.columns.tolist(), key="pd_pivot_columns")
                
                numeric_cols_pivot = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols_pivot:
                    st.warning("No numeric columns available for 'Values' in Pivot Table.")
                    values_col_pivot = None
                else:
                    values_col_pivot = st.selectbox("Values (Numeric Column)", numeric_cols_pivot, key="pd_pivot_values")
                
                agg_func_pivot = st.selectbox("Aggregation Function", ["mean", "sum", "count", "min", "max", "std"], key="pd_pivot_aggfunc")
                fill_value_pivot = st.number_input("Fill Value for Missing Data", value=0, key="pd_pivot_fill")

                if st.button("Create Pivot Table", key="pd_execute_pivot"):
                    if not index_cols and not columns_cols:
                        st.error("Please select at least one Index or Columns field.")
                    elif not values_col_pivot and agg_func_pivot not in ['count']: # Count can work without specific values col if index/columns are set
                        st.error("Please select a 'Values' column for aggregations other than 'count'.")
                    else:
                        try:
                            # If 'count' is used and no values_col_pivot, pandas might count occurrences based on index/columns.
                            # If values_col_pivot is None for 'count', it counts non-NA entries for the combinations.
                            # If aggfunc is 'count' and values_col_pivot is specified, it counts non-NA in that column.
                            
                            pivot_df = pd.pivot_table(
                                df,
                                index=index_cols if index_cols else None,
                                columns=columns_cols if columns_cols else None,
                                values=values_col_pivot if values_col_pivot else None, # Pass None if not selected
                                aggfunc=agg_func_pivot,
                                fill_value=fill_value_pivot
                            )
                            st.success("Pivot Table created successfully.")
                            st.dataframe(pivot_df)
                            st.session_state.df_pivot_temp = pivot_df
                        except Exception as e:
                            st.error(f"Error creating Pivot Table: {str(e)}")

        elif quick_ops == "Window Functions":
            st.subheader("🪟 Window Functions (Rolling/Expanding)")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                numeric_cols_window = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols_window:
                    st.warning("No numeric columns available for Window Functions.")
                else:
                    value_col_window = st.selectbox("Select Column for Window Function", numeric_cols_window, key="pd_window_value_col")
                    window_type = st.radio("Window Type", ["Rolling", "Expanding"], key="pd_window_type")
                    
                    min_periods_window = st.number_input("Minimum Periods", min_value=1, value=1, step=1, key="pd_window_min_periods")
                    
                    if window_type == "Rolling":
                        window_size = st.number_input("Window Size (for Rolling)", min_value=1, value=3, step=1, key="pd_window_size")
                        center_window = st.checkbox("Center Window (for Rolling)", value=False, key="pd_window_center")
                    
                    agg_func_window = st.selectbox(
                        "Aggregation Function", 
                        ["mean", "sum", "std", "count", "min", "max", "median", "var", "skew", "kurt"], 
                        key="pd_window_agg_func"
                    )
                    new_col_name_window = st.text_input("New Column Name", value=f"{value_col_window}_{window_type.lower()}_{agg_func_window}", key="pd_window_new_col_name")

                    if st.button("Apply Window Function", key="pd_execute_window"):
                        if not new_col_name_window.strip():
                            st.error("Please provide a name for the new column.")
                        else:
                            try:
                                df_copy = df.copy()
                                if window_type == "Rolling":
                                    series_window = df_copy[value_col_window].rolling(window=window_size, min_periods=min_periods_window, center=center_window)
                                else: # Expanding
                                    series_window = df_copy[value_col_window].expanding(min_periods=min_periods_window)
                                
                                # Apply aggregation
                                df_copy[new_col_name_window] = getattr(series_window, agg_func_window)()
                                
                                st.success(f"Applied {window_type} {agg_func_window} to '{value_col_window}', new column '{new_col_name_window}' created.")
                                st.dataframe(df_copy.head())
                                st.session_state.df = df_copy # Update main DataFrame
                            except Exception as e:
                                st.error(f"Error applying window function: {str(e)}")
        
        elif quick_ops == "Melt":
            st.subheader("🍦 Melt DataFrame")
            if df.empty:
                st.warning("DataFrame is empty.")
            else:
                id_vars = st.multiselect("Select ID Columns (to keep)", df.columns.tolist(), key="pd_melt_id_vars")
                value_vars_options = [col for col in df.columns if col not in id_vars]
                value_vars = st.multiselect("Select Value Columns (to unpivot)", value_vars_options, default=value_vars_options, key="pd_melt_value_vars")
                
                var_name = st.text_input("New Variable Column Name:", value="variable", key="pd_melt_var_name")
                value_name = st.text_input("New Value Column Name:", value="value", key="pd_melt_value_name")
            
                if st.button("Execute Melt", key="pd_execute_melt"):
                    if not id_vars and not value_vars:
                        st.warning("Melting without id_vars or value_vars will unpivot all columns. This might not be intended.")
                    if not var_name.strip() or not value_name.strip():
                        st.error("Please provide names for the new variable and value columns.")
                    else:
                        try:
                            melted_df = df.melt(
                                id_vars=id_vars if id_vars else None, # Pass None if empty, pandas handles it
                                value_vars=value_vars if value_vars else None, # Pass None if empty
                                var_name=var_name,
                                value_name=value_name
                            )
                            st.success(f"Melted DataFrame created with {len(melted_df)} rows.")
                            st.dataframe(melted_df.head())
                            st.session_state.df_melt_temp = melted_df
                        except Exception as e:
                            st.error(f"Error melting DataFrame: {str(e)}")
        
        elif quick_ops == "Query":
            st.subheader("🔍 DataFrame Query")
            st.info("Use pandas query syntax. Example: column_name > 100 & other_column == 'value'")
            
            query_str = st.text_input("Query Expression:", "", key="pd_query_str")
            
            if query_str and st.button("Execute Query", key="pd_execute_df_query"):
                try:
                    result = df.query(query_str)
                    st.success(f"Query returned {len(result)} rows")
                    st.dataframe(result)
                except Exception as e:
                    st.error(f"Query Error: {str(e)}")
        
        elif quick_ops == "Apply Function":
            st.subheader("✨ Apply Function to Column")
            apply_col = st.selectbox("Select Column", df.columns.tolist(), key="pd_apply_col")
            st.warning("⚠️ Using `eval()` for custom functions can be a security risk if the input is not trusted. Use with caution.")
            st.info("Enter a Python function string. Use 'x' as the element placeholder. Example: `lambda x: x * 2` or `lambda x: str(x).upper()`")
            function_str = st.text_input("Function (e.g., lambda x: x * 2):", key="pd_apply_func_str")
            new_col_name_apply = st.text_input("New Column Name (optional, if blank, column is modified in-place for preview):", key="pd_apply_new_col_name")
            
            if apply_col and function_str and st.button("Execute Apply", key="pd_execute_apply"):
                try:
                    # Safely evaluate the function string
                    func = eval(function_str)
                    
                    if new_col_name_apply.strip():
                        df[new_col_name_apply] = df[apply_col].apply(func)
                        st.success(f"Applied function to '{apply_col}' and created '{new_col_name_apply}'.")
                        st.dataframe(df[[apply_col, new_col_name_apply]].head())
                        st.session_state.df = df # Update session state
                    else:
                        # Apply in place or show result without adding column
                        result = df[apply_col].apply(func)
                        st.success(f"Applied function to '{apply_col}'. Result preview:")
                        st.dataframe(result.head().to_frame(name=f"{apply_col}_transformed"))
                        
                except Exception as e:
                    st.error(f"Error applying function: {str(e)}")
        
        elif quick_ops == "Custom":
            st.subheader("🛠️ Custom Pandas Code")
            
            pandas_code = st.text_area(
                "Pandas Code:",
                value="# Use 'df' to access your data\nresult = df.head()\nprint(result)",
                height=200,
                key="pd_custom_code_area"
            )
            
            if st.button("Execute Pandas Code", key="pd_execute_custom_code"):
                try:
                    exec_globals = {'df': df, 'pd': pd, 'np': np}
                    exec(pandas_code, exec_globals)
                    
                    if 'result' in exec_globals:
                        st.subheader("Results:")
                        if isinstance(exec_globals['result'], pd.DataFrame):
                            st.dataframe(exec_globals['result'])
                        else:
                            st.write(exec_globals['result'])
                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")

elif selected_tool == "🌐 Web Scraping Tool":
    st.markdown('<h2 class="tool-header">🌐 Advanced Web Scraping Tool</h2>', unsafe_allow_html=True)
    
    st.subheader("🔗 URL and Element Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = st.text_input("🌐 Enter URL:", placeholder="https://example.com")
        
        # Scraping method
        method = st.selectbox("Scraping Method", [
            "Extract Text by Tag",
            "Extract by CSS Selector", 
            "Extract Table Data",
            "Extract All Links",
            "Extract Images",
            "Custom BeautifulSoup"
        ])
        
        if method == "Extract Text by Tag":
            tag = st.text_input("HTML Tag:", value="p")
            class_name = st.text_input("CSS Class (optional):")
            limit = st.number_input("Limit results:", min_value=1, max_value=100, value=10)
            
        elif method == "Extract by CSS Selector":
            selector = st.text_input("CSS Selector:", placeholder="div.content p")
            limit = st.number_input("Limit results:", min_value=1, max_value=100, value=10)
            
        elif method == "Extract Table Data":
            table_index = st.number_input("Table Index (0-based):", min_value=0, value=0)
            
        elif method == "Extract All Links":
            filter_text = st.text_input("Filter links containing text (optional):")
            
        elif method == "Extract Images":
            min_width = st.number_input("Minimum width (optional):", min_value=0, value=0)
            
        elif method == "Custom BeautifulSoup":
            custom_code = st.text_area(
                "Custom BeautifulSoup Code:",
                value="""# Use 'soup' variable to access parsed HTML
# Example:
titles = soup.find_all('h1')
for title in titles:
    print(title.get_text())""",
                height=150
            )
    
    with col2:
        st.subheader("⚙️ Advanced Settings")
        
        headers = st.checkbox("Use Custom Headers")
        if headers:
            user_agent = st.text_input("User Agent:", value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        delay = st.slider("Delay between requests (seconds):", 0.0, 5.0, 1.0)
        timeout = st.slider("Request timeout (seconds):", 5, 30, 10)
        
        # Export options
        export_format = st.selectbox("Export Format", ["CSV", "JSON", "TXT"])
    
    if st.button("🚀 Start Scraping", type="primary"):
        if not url:
            st.error("Please enter a URL!")
        else:
            # Build params dict explicitly for robustness
            scrape_params = {
                'headers': headers,
                'delay': delay,
                'timeout': timeout
            }
            if headers:
                scrape_params['user_agent'] = user_agent
            
            if method == "Extract Text by Tag":
                scrape_params.update({'tag': tag, 'class_name': class_name, 'limit': limit})
            elif method == "Extract by CSS Selector":
                scrape_params.update({'selector': selector, 'limit': limit})
            elif method == "Extract Table Data":
                scrape_params.update({'table_index': table_index})
            elif method == "Extract All Links":
                scrape_params.update({'filter_text': filter_text})
            elif method == "Extract Images":
                scrape_params.update({'min_width': min_width})
            elif method == "Custom BeautifulSoup":
                scrape_params.update({'custom_code': custom_code})

            scrape_website(url, method, scrape_params, export_format)

# --- AI-Powered Insights (Gemini) ---
if selected_tool == "🤖 AI-Powered Insights (Gemini)":
    st.markdown('<h2 class="tool-header">🤖 AI-Powered Insights (Gemini)</h2>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload data first to use AI insights!")
    elif not st.session_state.gemini_model:
        st.warning("Please configure your Google AI API Key in the sidebar to enable this tool.")
    else:
        df = st.session_state.df
        
        ai_options = [
            "📝 Automated Data Summary",
            "📊 Natural Language to Chart",
            "🛠️ Feature Engineering Suggestions",
            "💬 Natural Language to Code",
            "🧠 Code Explanation"
        ]
        selected_ai_task = st.selectbox("Select an AI Task", ai_options)

        if selected_ai_task == "📝 Automated Data Summary":
            st.subheader("📝 Automated Data Summary")
            if st.button("Generate Summary", type="primary"):
                # Create a comprehensive prompt
                with io.StringIO() as buffer:
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                
                prompt = f"""
You are an expert data analyst. Analyze the following dataset and provide a concise summary.
The dataset has {df.shape[0]} rows and {df.shape[1]} columns.

Here is the output of `df.info()`:
```
{info_str}
```

Here is the output of `df.describe()` for numeric columns:
```
{df.describe(include=np.number).to_string()}
```

Here is the output of `df.describe()` for categorical columns:
```
{df.describe(include=['object', 'category']).to_string()}
```

Based on this information, provide a summary covering:
1.  **Overall Structure**: Mention the number of rows, columns, and data types.
2.  **Key Numeric Insights**: Talk about the central tendency, spread, and potential outliers for important numeric columns.
3.  **Key Categorical Insights**: Discuss the distribution, cardinality, and most frequent categories for important object/categorical columns.
4.  **Data Quality Issues**: Point out any potential issues like missing values, high cardinality columns, or columns that might need type conversion.
5.  **Next Steps**: Suggest 2-3 potential next steps for analysis or feature engineering.

Keep the summary clear, concise, and use markdown for formatting.
"""
                summary = generate_gemini_content(prompt)
                if summary:
                    st.markdown(summary)

        elif selected_ai_task == "📊 Natural Language to Chart":
            st.subheader("📊 Natural Language to Chart")
            st.info("Describe the chart you want to create. For example: 'a scatter plot of column A vs column B, colored by column C'. The AI will attempt to fix its own code if it fails.")
            
            if 'chart_request' not in st.session_state:
                st.session_state.chart_request = "a bar chart showing the average income per city"
            
            chart_request = st.text_area("Your chart request:", key="chart_request")
            
            if st.button("Generate Chart", type="primary"):
                if chart_request:
                    with st.spinner("Generating initial code..."):
                        # Get schema for the prompt
                        schema = pd.DataFrame({
                            'Column': df.columns,
                            'DataType': df.dtypes.astype(str)
                        }).to_string()

                        initial_prompt = f"""
You are an expert Python data visualization specialist who uses the plotly.express library.
Given a pandas DataFrame named `df` with the following schema:
{schema}

And a small sample of the data:
{df.head().to_string()}

Write Python code using `plotly.express` (as px) to generate a chart that fulfills the following user request.
"{chart_request}"

Your code MUST:
1.  Import `plotly.express` as `px`.
2.  Create a single figure object and assign it to a variable named `fig`.
3.  Do NOT use `fig.show()` or `st.plotly_chart()`.
4.  Provide ONLY the Python code in a single code block, without any explanation or surrounding text.
5.  Before plotting, ensure the necessary columns are of the correct data type (e.g., using `pd.to_numeric` or `pd.to_datetime`). Handle potential errors in conversion (e.g., `errors='coerce'`).
6.  Handle potential missing values in the columns used for plotting (e.g., by using `.dropna()`). This is very important.
"""
                        
                        generated_code = generate_gemini_content(initial_prompt)
                    
                    if generated_code:
                        # Clean up the response to get only the code block
                        cleaned_code = re.sub(r"```(python)?\n", "", generated_code)
                        cleaned_code = re.sub(r"```", "", cleaned_code).strip()
                        
                        st.subheader("Generated Code")
                        st.code(cleaned_code, language='python')
                        
                        st.subheader("Generated Chart")
                        st.warning("⚠️ Executing AI-generated code. Review the code for safety before running in production.")
                        
                        max_retries = 1
                        for attempt in range(max_retries + 1):
                            try:
                                # Create a safe execution environment
                                exec_globals = {
                                    'df': df.copy(),
                                    'pd': pd,
                                    'px': px,
                                    'go': go,
                                    'np': np,
                                    'nan': np.nan # Fix for 'nan' is not defined error
                                }
                                
                                # Execute code
                                exec(cleaned_code, exec_globals)
                                
                                # Capture the figure
                                if 'fig' in exec_globals and isinstance(exec_globals['fig'], go.Figure):
                                    fig = exec_globals['fig']
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.success("Chart generated successfully!")
                                    break # Success, exit the loop
                                else:
                                    # This is a logical error, not a syntax/runtime one. We'll trigger a retry.
                                    raise ValueError("The generated code did not create a 'fig' variable of type go.Figure.")

                            except Exception as e:
                                st.error(f"Attempt {attempt + 1} failed: {e}")
                                if attempt < max_retries:
                                    st.info("🤖 The code failed. Asking Gemini for a fix...")
                                    with st.spinner("Attempting to self-correct the code..."):
                                        fix_prompt = f"""You are a Python debugging expert. The following code, intended to generate a Plotly chart, failed with an error.
Original user request: "{chart_request}"
Faulty Code:\n```python\n{cleaned_code}\n```\nError Message:\n```\n{e}\n```
Please fix the code. Your response should only contain the corrected Python code in a single code block. Do not add any explanation. Ensure the corrected code defines a plotly figure named `fig`."""
                                        corrected_code_response = generate_gemini_content(fix_prompt)
                                        if corrected_code_response:
                                            cleaned_code = re.sub(r"```(python)?\n", "", corrected_code_response)
                                            cleaned_code = re.sub(r"```", "", cleaned_code).strip()
                                            st.subheader(f"Corrected Code (Attempt {attempt + 2})")
                                            st.code(cleaned_code, language='python')
                                        else:
                                            st.error("Could not get a fix from Gemini. Stopping.")
                                            break # Break if Gemini fails to provide a fix
                                else:
                                    st.error("Failed to generate a working chart after retries.")
                    else:
                        st.error("Failed to generate code from the initial prompt.")
                else:
                    st.warning("Please enter a chart request.")

        elif selected_ai_task == "🛠️ Feature Engineering Suggestions":
            st.subheader("🛠️ Feature Engineering Suggestions")
            st.info("Let Gemini analyze your data and suggest new features to improve model performance or analysis.")
            
            if st.button("Generate Suggestions", type="primary"):
                # Create a comprehensive prompt
                with io.StringIO() as buffer:
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                
                prompt = f"""
You are an expert data scientist specializing in feature engineering.
Analyze the following dataset and suggest potential new features that could be created.

The dataset has {df.shape[0]} rows and {df.shape[1]} columns.

Here is the output of `df.info()`:
```
{info_str}
```

Here is a sample of the data (`df.head()`):
```
{df.head().to_string()}
```

Based on this information, provide a list of feature engineering suggestions. For each suggestion:
1.  **Clearly state the new feature(s)** to be created.
2.  **Provide the rationale**: Explain *why* this feature might be useful for analysis or a machine learning model.
3.  **Provide the Pandas code** to create the feature. Assume the DataFrame is named `df`.

Organize your response using markdown, with clear headings for each suggestion category (e.g., "Date/Time Features", "Interaction Features", "Binning/Discretization", "Categorical Encoding").
If no features of a certain type are applicable, don't include that section.
"""
                suggestions = generate_gemini_content(prompt)
                if suggestions:
                    st.markdown(suggestions)

        elif selected_ai_task == "💬 Natural Language to Code":
            st.subheader("💬 Natural Language to Code")
            
            code_type = st.radio("Select Code Type", ["SQL", "Pandas"], horizontal=True)
            
            query = st.text_area("What would you like to do with the data?", placeholder="e.g., 'Find the top 5 cities with the highest average income'")
            
            if st.button("Generate Code", type="primary"):
                if query:
                    # Get schema for the prompt
                    schema = pd.DataFrame({
                        'Column': df.columns,
                        'DataType': df.dtypes.astype(str)
                    }).to_string()

                    if code_type == "SQL":
                        prompt = f"You are an expert SQL developer.\nGiven a table named `data` with the following schema:\n{schema}\n\nWrite a SQL query to answer the following question:\n\"{query}\"\n\nProvide only the SQL code in a single code block, without any explanation."
                    else: # Pandas
                        prompt = f"You are an expert Python developer using pandas.\nGiven a pandas DataFrame named `df` with the following schema:\n{schema}\n\nWrite Python code using pandas to answer the following question:\n\"{query}\"\n\nAssume the DataFrame is already loaded in a variable named `df`.\nProvide only the Python code in a single code block, without any explanation."
                    
                    generated_code = generate_gemini_content(prompt)
                    if generated_code:
                        # Clean up the response to get only the code block
                        cleaned_code = re.sub(r"```(python|sql)?\n", "", generated_code)
                        cleaned_code = re.sub(r"```", "", cleaned_code).strip()
                        st.code(cleaned_code, language=code_type.lower())
                else:
                    st.warning("Please enter a query.")

        elif selected_ai_task == "🧠 Code Explanation":
            st.subheader("🧠 Code Explanation")
            code_to_explain = st.text_area("Paste your code here (SQL or Python)", height=200)
            
            if st.button("Explain Code", type="primary"):
                if code_to_explain:
                    prompt = f"You are an expert code reviewer and teacher.\nExplain the following code snippet step-by-step.\nDescribe what the code does, its purpose, and how it works.\nUse markdown for formatting.\n\nCode:\n```\n{code_to_explain}\n```"
                    explanation = generate_gemini_content(prompt)
                    if explanation:
                        st.markdown(explanation)
                else:
                    st.warning("Please paste some code to explain.")

# Ensure df is always available if it's in session state, for tools that might be selected before data upload interaction
if 'df' in st.session_state and st.session_state.df is not None and 'df' not in locals():
    df = st.session_state.df
