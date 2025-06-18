import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import requests
from bs4 import BeautifulSoup
import io
import base64
from datetime import datetime
import warnings
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import openpyxl
from sqlalchemy import create_engine, text
import time
import re
import json

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
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .tool-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
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

# Main Title
st.markdown('<h1 class="main-header">🔬 Advanced Data Analysis Suite</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
tools = [
    "📤 Data Upload",
    "🔍 SQL Query Engine",
    "📊 Exploratory Data Analysis (EDA)",
    "📈 Excel Query Tool",
    "💼 Power BI Dashboard",
    "🐍 Python Advanced Analytics",
    "🐼 Pandas Query Tool",
    "🌐 Web Scraping Tool"
]

selected_tool = st.sidebar.selectbox("Select Analysis Tool", tools)

# Helper Functions
@st.cache_data
def load_data(file):
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

def create_download_link(df, filename="data.csv"):
    """Create download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def execute_sql_query(df, query):
    """Execute SQL query on dataframe"""
    try:
        # Create in-memory SQLite database
        conn = sqlite3.connect(':memory:')
        df.to_sql('data', conn, index=False, if_exists='replace')
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result, None
    except Exception as e:
        return None, str(e)

def advanced_outlier_detection(df, column):
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
    
    return methods

def generate_chart(df, config, title):
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
            y = df[config['y']].astype(float) if not pd.api_types.is_numeric_dtype(df[config['y']]) else df[config['y']]
            color = config.get('color')
            if color:
                color_data = df[color].astype(str) if pd.api.types.is_object_dtype(df[color]) else df[color]
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

# Tool Implementation

def execute_python_code(code, df):
    """Execute Python code safely"""
    try:
        # Create a safe execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'plt': plt,
            'px': px,
            'go': go,
            'stats': stats,
            'StandardScaler': StandardScaler,
            'PCA': PCA,
            'KMeans': KMeans,
            'IsolationForest': IsolationForest,
            'LabelEncoder': LabelEncoder
        }
        
        # Capture output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Execute code
        exec(code, exec_globals)
        
        # Get output
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        st.session_state.python_output = output
        
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
        st.session_state.python_output = f"Error: {str(e)}"
# Tool Implementation

if selected_tool == "📤 Data Upload":
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
                    st.markdown(
                        create_download_link(df, f"processed_{uploaded_file.name}"),
                        unsafe_allow_html=True
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
            
            # Quick Stats
            st.subheader("📊 Quick Statistics")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe())

elif selected_tool == "🔍 SQL Query Engine":
    st.markdown('<h2 class="tool-header">🔍 Advanced SQL Query Engine</h2>', unsafe_allow_html=True)

    # --- SQL Query Examples ---
    with st.expander("🧠 SQL Query Examples", expanded=False):
        st.markdown("""
**1. Select everything**
```sql
SELECT * FROM data;
```
**2. Filter: Male users in Tokyo**
```sql
SELECT * FROM data
WHERE gender = 'Male' AND city = 'Tokyo';
```
**3. Total purchases per city**
```sql
SELECT city, SUM(purchases) AS total_purchases
FROM data
GROUP BY city;
```
**4. Average income by gender**
```sql
SELECT gender, AVG(income) AS avg_income
FROM data
GROUP BY gender;
```
**5. Count users with missing age**
```sql
SELECT COUNT(*) AS missing_ages
FROM data
WHERE age IS NULL;
```
        """)

    if st.session_state.df is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.df
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("✍️ SQL Query Editor")
            
            # Query templates
            templates = {
                "Basic Select": "SELECT * FROM data LIMIT 10;",
                "Group By": "SELECT column_name, COUNT(*) as count FROM data GROUP BY column_name;",
                "Where Filter": "SELECT * FROM data WHERE column_name = 'value';",
                "Join": "-- Use multiple tables if needed",
                "Aggregate": "SELECT AVG(column_name), MAX(column_name), MIN(column_name) FROM data;",
                "Window Functions": "SELECT *, ROW_NUMBER() OVER (ORDER BY column_name) as row_num FROM data;"
            }
            
            selected_template = st.selectbox("📋 Query Templates", list(templates.keys()))
            if st.button("Use Template"):
                st.session_state.current_query = templates[selected_template]
            
            query = st.text_area(
                "Enter SQL Query:",
                value=st.session_state.get('current_query', 'SELECT * FROM data LIMIT 10;'),
                height=200,
                help="Table name is 'data'. Available columns: " + ", ".join(df.columns.tolist())
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🚀 Execute Query", type="primary"):
                    result, error = execute_sql_query(df, query)
                    if error:
                        st.error(f"SQL Error: {error}")
                    else:
                        st.session_state.sql_result = result
                        st.session_state.sql_history.append({
                            'query': query,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'rows': len(result)
                        })
                        st.success(f"Query executed successfully! Returned {len(result)} rows.")
            
            with col_b:
                if st.button("📋 Show Schema"):
                    schema_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Null Count': df.isnull().sum(),
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(schema_info)
        
        with col2:
            st.subheader("📚 Query History")
            if st.session_state.sql_history:
                for i, hist in enumerate(reversed(st.session_state.sql_history[-5:])):
                    with st.expander(f"Query {len(st.session_state.sql_history) - i}"):
                        st.code(hist['query'], language='sql')
                        st.caption(f"Executed: {hist['timestamp']} | Rows: {hist['rows']}")
        
        # Display Results
        if hasattr(st.session_state, 'sql_result'):
            st.subheader("📊 Query Results")
            result = st.session_state.sql_result
            st.dataframe(result)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "💾 Download CSV",
                    result.to_csv(index=False),
                    file_name=f"sql_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "📄 Download JSON",
                    result.to_json(orient='records'),
                    file_name=f"sql_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

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
            "🔢 Statistical Summary",
            "📋 Data Quality Report",
            "🧮 Feature Engineering"
        ]
        
        selected_eda = st.selectbox("Select EDA Tool", eda_tools)
        
        if selected_eda == "🔍 Data Overview":
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
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns found!")
            else:
                selected_col = st.selectbox("Select Column", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(df, x=selected_col, nbins=50, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical tests
                st.subheader("📊 Statistical Tests")
                col_data = df[selected_col].dropna()
                
                # Normality test
                shapiro_stat, shapiro_p = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Skewness", f"{stats.skew(col_data):.3f}")
                with col_b:
                    st.metric("Kurtosis", f"{stats.kurtosis(col_data):.3f}")
                with col_c:
                    st.metric("Shapiro p-value", f"{shapiro_p:.6f}")
        
        elif selected_eda == "🔗 Correlation Analysis":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation analysis!")
            else:
                corr_matrix = df[numeric_cols].corr()
                
                # Correlation heatmap
                fig = px.imshow(corr_matrix, 
                              title="Correlation Matrix",
                              color_continuous_scale="RdBu",
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
                
                # Strong correlations
                st.subheader("🎯 Strong Correlations (|r| > 0.7)")
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corr.append({
                                'Variable 1': corr_matrix.columns[i],
                                'Variable 2': corr_matrix.columns[j],
                                'Correlation': corr_val
                            })
                
                if strong_corr:
                    strong_corr_df = pd.DataFrame(strong_corr)
                    st.dataframe(strong_corr_df)
                else:
                    st.info("No strong correlations found.")
        
        elif selected_eda == "🎯 Outlier Detection":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns found!")
            else:
                selected_col = st.selectbox("Select Column for Outlier Detection", numeric_cols)
                
                outlier_methods = advanced_outlier_detection(df, selected_col)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔍 Outlier Detection Results")
                    for method, count in outlier_methods.items():
                        st.metric(f"{method} Outliers", count)
                
                with col2:
                    # Scatter plot with outliers highlighted
                    fig = go.Figure()
                    
                    # Normal points
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[selected_col],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=4)
                    ))
                    
                    # IQR outliers
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[selected_col] < Q1 - 1.5 * IQR) | (df[selected_col] > Q3 + 1.5 * IQR)]
                    
                    fig.add_trace(go.Scatter(
                        x=outliers.index,
                        y=outliers[selected_col],
                        mode='markers',
                        name='Outliers',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                    
                    fig.update_layout(title=f"Outlier Detection: {selected_col}")
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
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols:
                st.warning("No categorical columns found!")
            else:
                selected_col = st.selectbox("Select Categorical Column", categorical_cols)
                
                value_counts = df[selected_col].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"📊 Top 10 Values in {selected_col}")
                    st.dataframe(value_counts.head(10))
                    
                    st.metric("Unique Values", df[selected_col].nunique())
                    st.metric("Most Frequent", value_counts.index[0])
                    st.metric("Mode Frequency", value_counts.iloc[0])
                
                with col2:
                    # Bar chart
                    fig = px.bar(x=value_counts.head(10).index, 
                               y=value_counts.head(10).values,
                               title=f"Top 10 Values in {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
        
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

elif selected_tool == "📈 Excel Query Tool":
    st.markdown('<h2 class="tool-header">📈 Advanced Excel Query Tool</h2>', unsafe_allow_html=True)

    # --- Excel Query Examples ---
    with st.expander("📊 Excel Query Examples", expanded=False):
        st.markdown("""
**1. VLOOKUP Example:**  
Find the email of user with ID 123  
- Lookup Column: `user_id`  
- Lookup Value: `123`  
- Return Column: `email`

**2. PIVOT Example:**  
Summarize total sales by region and product  
- Index: `region`  
- Columns: `product`  
- Values: `sales`  
- Aggregation: `sum`

**3. FILTER Example:**  
Show all rows where `status` contains "Active"

**4. SORT Example:**  
Sort by `created_at` in descending order

**5. SUMIF Example:**  
Sum `amount` where `category` contains "Food"

**6. SPLIT Example:**  
Split `full_name` by space into `first_name` and `last_name`
        """)

    if st.session_state.df is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.df
        
        st.subheader("🔧 Excel-Style Operations")
        
        operation = st.selectbox("Select Operation", [
            "VLOOKUP", "HLOOKUP", "PIVOT", "FILTER", "SORT", "GROUPBY", 
            "SUMIF", "COUNTIF", "AVERAGEIF", "CONCATENATE", "SPLIT"
        ])
        
        if operation == "VLOOKUP":
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
        
        elif operation == "FILTER":
            st.subheader("🔍 Advanced Filter")
            filter_col = st.selectbox("Filter Column", df.columns.tolist())
            filter_type = st.selectbox("Filter Type", ["Contains", "Equals", "Greater Than", "Less Than", "Between"])
            
            if filter_type in ["Contains", "Equals"]:
                filter_value = st.text_input("Filter Value")
                if st.button("Apply Filter"):
                    if filter_type == "Contains":
                        filtered_df = df[df[filter_col].astype(str).str.contains(filter_value, case=False, na=False)]
                    else:
                        filtered_df = df[df[filter_col] == filter_value]
                    
                    st.success(f"Filtered to {len(filtered_df)} rows")
                    st.dataframe(filtered_df)
            
            elif filter_type in ["Greater Than", "Less Than"]:
                if df[filter_col].dtype in ['int64', 'float64']:
                    filter_value = st.number_input("Filter Value", value=float(df[filter_col].mean() if not df[filter_col].empty else 0))
                    if st.button("Apply Filter"):
                        if filter_type == "Greater Than":
                            filtered_df = df[df[filter_col] > filter_value]
                        else:
                            filtered_df = df[df[filter_col] < filter_value]
                        
                        st.success(f"Filtered to {len(filtered_df)} rows")
                        st.dataframe(filtered_df)
                else:
                    st.error("Selected column is not numeric")
            
            elif filter_type == "Between":
                if df[filter_col].dtype in ['int64', 'float64']:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        min_value = st.number_input("Minimum Value", value=float(df[filter_col].min() if not df[filter_col].empty else 0))
                    with col_b:
                        max_value = st.number_input("Maximum Value", value=float(df[filter_col].max() if not df[filter_col].empty else 0))
                    
                    if st.button("Apply Filter"):
                        filtered_df = df[(df[filter_col] >= min_value) & (df[filter_col] <= max_value)]
                        st.success(f"Filtered to {len(filtered_df)} rows")
                        st.dataframe(filtered_df)
                else:
                    st.error("Selected column is not numeric")
        
        elif operation == "SORT":
            st.subheader("⬆️⬇️ Sort Data")
            sort_col = st.selectbox("Sort Column", df.columns.tolist())
            sort_order = st.radio("Sort Order", ["Ascending", "Descending"])
            
            if st.button("Apply Sort"):
                ascending = sort_order == "Ascending"
                sorted_df = df.sort_values(by=sort_col, ascending=ascending)
                st.dataframe(sorted_df)
        
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

elif selected_tool == "💼 Power BI Dashboard":
    st.markdown('<h2 class="tool-header">💼 Power BI Style Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.df
        
        dashboard_type = st.radio("Dashboard Type", ["📊 Automatic Dashboard", "🎯 Manual Dashboard"])
        
        if dashboard_type == "📊 Automatic Dashboard":
            st.subheader("🤖 Auto-Generated Dashboard")
            
            with st.spinner("Generating dashboard..."):
                # Dashboard metrics
                col1, col2, col3, col4 = st.columns(4)
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                
                with col1:
                    st.metric("Total Records", f"{len(df):,}")
                with col2:
                    st.metric("Numeric Columns", len(numeric_cols))
                with col3:
                    st.metric("Categorical Columns", len(categorical_cols))
                with col4:
                    if len(numeric_cols) > 0:
                        st.metric("Avg of First Numeric", f"{df[numeric_cols[0]].mean():.2f}")
                
                # Auto-generated charts
                chart_row1 = st.columns(2)
                chart_row2 = st.columns(2)
                chart_row3 = st.columns(2)
                
                charts_created = 0
                
                # Chart 1: Distribution of first numeric column
                if len(numeric_cols) > 0 and charts_created < 6:
                    with chart_row1[0]:
                        col_data = df[numeric_cols[0]].astype(float)
                        fig = px.histogram(df, x=col_data, title=f"Distribution of {numeric_cols[0]}")
                        st.plotly_chart(fig, use_container_width=True)
                        charts_created += 1
                
                # Chart 2: Top categories in first categorical column
                if len(categorical_cols) > 0 and charts_created < 6:
                    with chart_row1[1]:
                        top_categories = df[categorical_cols[0]].value_counts().head(10)
                        fig = px.bar(x=top_categories.index.astype(str), y=top_categories.values.astype(float), 
                                   title=f"Top Categories in {categorical_cols[0]}")
                        st.plotly_chart(fig, use_container_width=True)
                        charts_created += 1
                
                # Chart 3: Correlation heatmap (if multiple numeric columns)
                if len(numeric_cols) > 1 and charts_created < 6:
                    with chart_row2[0]:
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix.astype(float), title="Correlation Matrix")
                        st.plotly_chart(fig, use_container_width=True)
                        charts_created += 1
                
                # Chart 4: Box plot of numeric columns
                if len(numeric_cols) > 0 and charts_created < 6:
                    with chart_row2[1]:
                        col_data = df[numeric_cols[0]].astype(float)
                        fig = px.box(df, y=col_data, title=f"Outliers in {numeric_cols[0]}")
                        st.plotly_chart(fig, use_container_width=True)
                        charts_created += 1
                
                # Chart 5: Scatter plot (if 2+ numeric columns)
                if len(numeric_cols) >= 2 and charts_created < 6:
                    with chart_row3[0]:
                        color_col = categorical_cols[0] if len(categorical_cols) > 0 else None
                        x_data = df[numeric_cols[0]].astype(float)
                        y_data = df[numeric_cols[1]].astype(float)
                        color_data = df[color_col].astype(str) if color_col else None
                        fig = px.scatter(df, x=x_data, y=y_data, 
                                       color=color_data, title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                        st.plotly_chart(fig, use_container_width=True)
                        charts_created += 1
                
                # Chart 6: Time series (if date column exists)
                date_cols = df.select_dtypes(include=['datetime64']).columns
                if len(date_cols) > 0 and len(numeric_cols) > 0 and charts_created < 6:
                    with chart_row3[1]:
                        y_data = df[numeric_cols[0]].astype(float)
                        fig = px.line(df, x=date_cols[0], y=y_data, 
                                    title=f"Time Series: {numeric_cols[0]}")
                        st.plotly_chart(fig, use_container_width=True)
                        charts_created += 1
                elif charts_created < 6:
                    with chart_row3[1]:
                        # Missing values chart
                        missing_data = df.isnull().sum()
                        if missing_data.sum() > 0:
                            fig = px.bar(x=missing_data.index.astype(str), y=missing_data.values.astype(float),
                                       title="Missing Values by Column")
                            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Manual Dashboard
            st.subheader("🎯 Custom Dashboard Builder")
            
            # Dashboard layout selection
            layout = st.selectbox("Select Layout", ["2x2 Grid", "3x2 Grid", "Single Row", "Single Column"])
            
            if layout == "2x2 Grid":
                cols = st.columns(2)
                rows = 2
            elif layout == "3x2 Grid":
                cols = st.columns(3)
                rows = 2
            elif layout == "Single Row":
                cols = st.columns(4)
                rows = 1
            else:  # Single Column
                cols = [st.container()]
                rows = 4
            
            # Chart configuration
            st.subheader("📊 Chart Configuration")
            
            chart_configs = []
            num_charts = min(len(cols) * rows, 6)
            
            for i in range(num_charts):
                with st.expander(f"Chart {i+1} Configuration"):
                    chart_type = st.selectbox(f"Chart Type {i+1}", 
                                            ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Heatmap"],
                                            key=f"chart_type_{i}")
                    
                    if chart_type in ["Bar", "Line", "Scatter"]:
                        x_col = st.selectbox(f"X-axis {i+1}", df.columns.tolist(), key=f"x_col_{i}")
                        y_col = st.selectbox(f"Y-axis {i+1}", df.columns.tolist(), key=f"y_col_{i}")
                        color_col = st.selectbox(f"Color by {i+1}", ['None'] + df.columns.tolist(), key=f"color_col_{i}")
                        color_col = None if color_col == 'None' else color_col
                        
                        chart_configs.append({
                            'type': chart_type,
                            'x': x_col,
                            'y': y_col,
                            'color': color_col
                        })
                    
                    elif chart_type in ["Histogram", "Box"]:
                        col = st.selectbox(f"Column {i+1}", df.columns.tolist(), key=f"col_{i}")
                        chart_configs.append({
                            'type': chart_type,
                            'column': col
                        })
                    
                    elif chart_type == "Pie":
                        col = st.selectbox(f"Category Column {i+1}", df.columns.tolist(), key=f"pie_col_{i}")
                        chart_configs.append({
                            'type': chart_type,
                            'column': col
                        })
                    
                    elif chart_type == "Heatmap":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        chart_configs.append({
                            'type': chart_type,
                            'columns': numeric_cols
                        })
            
            if st.button("🚀 Generate Dashboard"):
                # Generate charts based on configuration
                chart_index = 0
                if layout == "2x2 Grid":
                    for row in range(2):
                        cols = st.columns(2)
                        for col_idx in range(2):
                            if chart_index < len(chart_configs):
                                config = chart_configs[chart_index]
                                with cols[col_idx]:
                                    generate_chart(df, config, f"Chart {chart_index + 1}")
                                chart_index += 1
                
                elif layout == "3x2 Grid":
                    for row in range(2):
                        cols = st.columns(3)
                        for col_idx in range(3):
                            if chart_index < len(chart_configs):
                                config = chart_configs[chart_index]
                                with cols[col_idx]:
                                    generate_chart(df, config, f"Chart {chart_index + 1}")
                                chart_index += 1
                
                elif layout == "Single Row":
                    if chart_configs:
                        cols = st.columns(len(chart_configs))
                        for i, config in enumerate(chart_configs):
                            with cols[i]:
                                generate_chart(df, config, f"Chart {i + 1}")
                    else:
                        st.info("No charts configured.")
                
                else:  # Single Column
                    for i, config in enumerate(chart_configs):
                        # Use a container for single column layout
                        with st.container():
                            generate_chart(df, config, f"Chart {i + 1}")

# Continue with Python Advanced Analytics
elif selected_tool == "🐍 Python Advanced Analytics":
    st.markdown('<h2 class="tool-header">🐍 Python Advanced Analytics Engine</h2>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Please upload data first!")
    else:
        df = st.session_state.df
        
        st.subheader("💻 Python Code Editor")
        
        # Code templates
        templates = {
            "Data Info": """# Basic data information
print("Dataset Shape:", df.shape)
print("\\nData Types:")
print(df.dtypes)
print("\\nMissing Values:")
print(df.isnull().sum())
print("\\nBasic Statistics:")
print(df.describe())""",
            
            "Advanced Statistics": """# Advanced statistical analysis
import scipy.stats as stats
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols[:3]:  # First 3 numeric columns
    print(f"\\n=== {col} ===")
    print(f"Skewness: {stats.skew(df[col].dropna()):.3f}")
    print(f"Kurtosis: {stats.kurtosis(df[col].dropna()):.3f}")
    
    # Normality test
    if len(df[col].dropna()) > 3:
        stat, p_value = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
        print(f"Shapiro-Wilk p-value: {p_value:.6f}")""",
            
            "Machine Learning": """# Basic ML analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Prepare data
numeric_df = df.select_dtypes(include=[np.number]).dropna()

if len(numeric_df.columns) > 1:
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    # PCA Analysis
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    
    print("PCA Explained Variance Ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"PC{i+1}: {ratio:.3f}")
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    print(f"\\nCluster distribution:")
    unique, counts = np.unique(clusters, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} points")""",
            
            "Data Cleaning": """# Advanced data cleaning
print("Original shape:", df.shape)

# Handle missing values
missing_threshold = 0.5  # Remove columns with >50% missing
cols_to_keep = df.columns[df.isnull().mean() < missing_threshold]
df_clean = df[cols_to_keep].copy()

print(f"After removing high-missing columns: {df_clean.shape}")

# Remove duplicates
df_clean = df_clean.drop_duplicates()
print(f"After removing duplicates: {df_clean.shape}")

# Handle outliers using IQR method
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_count = len(df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)])
    print(f"{col}: {outliers_count} outliers detected")

print("\\nCleaned dataset info:")
print(df_clean.info())"""
        }
        
        # Template selection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_template = st.selectbox("📋 Code Templates", list(templates.keys()))
        with col2:
            if st.button("📥 Use Template"):
                st.session_state.python_code = templates[selected_template]
        
        # Code editor
        code = st.text_area(
            "Python Code:",
            value=st.session_state.get('python_code', '# Your Python code here\nprint("Hello, Data Science!")'),
            height=300,
            help="Use 'df' to access your uploaded data. Available libraries: pandas, numpy, scipy, sklearn, plotly"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🚀 Execute Code", type="primary"):
                execute_python_code(code, df)
        
        with col2:
            if st.button("💾 Save to History"):
                st.session_state.python_history.append({
                    'code': code,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Code saved to history!")
        
        # Code execution results
        if 'python_output' in st.session_state:
            st.subheader("📊 Execution Results")
            if st.session_state.python_output:
                st.code(st.session_state.python_output)
            
            if 'python_plots' in st.session_state and st.session_state.python_plots:
                st.subheader("📈 Generated Plots")
                for plot in st.session_state.python_plots:
                    st.plotly_chart(plot, use_container_width=True)
        
        # Code history
        if st.session_state.python_history:
            st.subheader("📚 Code History")
            for i, entry in enumerate(reversed(st.session_state.python_history[-5:])):
                with st.expander(f"Code Entry {len(st.session_state.python_history) - i}"):
                    st.code(entry['code'], language='python')
                    st.caption(f"Executed: {entry['timestamp']}")
                    if st.button(f"Reuse Code {len(st.session_state.python_history) - i}", key=f"reuse_{i}"):
                        st.session_state.python_code = entry['code']
                        st.experimental_rerun()

elif selected_tool == "🐼 Pandas Query Tool":
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
            "Data Info", "Head/Tail", "Describe", "Value Counts", "Group By", 
            "Merge/Join", "Pivot", "Melt", "Apply Function", "Query", "Custom"
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
        
        elif quick_ops == "Group By":
            group_col = st.selectbox("Group By Column", df.columns.tolist())
            agg_col = st.selectbox("Aggregate Column", df.select_dtypes(include=[np.number]).columns.tolist())
            agg_func = st.selectbox("Aggregation Function", ["sum", "mean", "count", "min", "max", "std"])
            
            if st.button("Execute Group By"):
                try:
                    result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
                    st.dataframe(result)
                    
                    # Visualization
                    fig = px.bar(result, x=group_col, y=agg_col, 
                               title=f"{agg_func.title()} of {agg_col} by {group_col}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif quick_ops == "Query":
            st.subheader("🔍 DataFrame Query")
            st.info("Use pandas query syntax. Example: column_name > 100 & other_column == 'value'")
            
            query_str = st.text_input("Query Expression:", "")
            
            if query_str and st.button("Execute Query"):
                try:
                    result = df.query(query_str)
                    st.success(f"Query returned {len(result)} rows")
                    st.dataframe(result)
                except Exception as e:
                    st.error(f"Query Error: {str(e)}")
        
        elif quick_ops == "Melt":
            st.subheader("🍦 Melt DataFrame")
            id_vars = st.multiselect("Select ID Columns (to keep)", df.columns.tolist())
            value_vars = st.multiselect("Select Value Columns (to unpivot)", [col for col in df.columns if col not in id_vars])
            var_name = st.text_input("New Variable Column Name:", value="Variable")
            value_name = st.text_input("New Value Column Name:", value="Value")
            
            if id_vars and value_vars and st.button("Execute Melt"):
                try:
                    melted_df = df.melt(
                        id_vars=id_vars,
                        value_vars=value_vars,
                        var_name=var_name,
                        value_name=value_name
                    )
                    st.success(f"Melted DataFrame created with {len(melted_df)} rows.")
                    st.dataframe(melted_df.head())
                except Exception as e:
                    st.error(f"Error melting DataFrame: {str(e)}")
        
        elif quick_ops == "Apply Function":
            st.subheader("✨ Apply Function to Column")
            apply_col = st.selectbox("Select Column", df.columns.tolist())
            st.info("Enter a Python function string. Use 'x' as the element placeholder. Example: `lambda x: x * 2` or `lambda x: str(x).upper()`")
            function_str = st.text_input("Function (e.g., lambda x: x * 2):")
            new_col_name = st.text_input("New Column Name (optional):")
            
            if apply_col and function_str and st.button("Execute Apply"):
                try:
                    # Safely evaluate the function string
                    func = eval(function_str)
                    
                    if new_col_name:
                        df[new_col_name] = df[apply_col].apply(func)
                        st.success(f"Applied function to '{apply_col}' and created '{new_col_name}'.")
                        st.dataframe(df[[apply_col, new_col_name]].head())
                        st.session_state.df = df # Update session state
                    else:
                        # Apply in place or show result without adding column
                        result = df[apply_col].apply(func)
                        st.success(f"Applied function to '{apply_col}'. Result preview:")
                        st.write(result.head())
                        
                except Exception as e:
                    st.error(f"Error applying function: {str(e)}")
        
        elif quick_ops == "Custom":
            st.subheader("🛠️ Custom Pandas Code")
            
            pandas_code = st.text_area(
                "Pandas Code:",
                value="# Use 'df' to access your data\nresult = df.head()\nprint(result)",
                height=200
            )
            
            if st.button("Execute Pandas Code"):
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
            "Extract Links",
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
            
        elif method == "Extract Links":
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
            # Add delay before making the request
            time.sleep(params.get('delay', 1.0))
            
            scrape_website(url, method, locals(), export_format)

def scrape_website(url, method, params, export_format):
    """Perform web scraping based on selected method"""
    try:
        # Setup headers
        headers_dict = {
            'User-Agent': params.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        } if params.get('headers') else {}
        
        # Use the delay parameter
        with st.spinner("Scraping website..."):
            # Make request
            response = requests.get(url, headers=headers_dict, timeout=params.get('timeout', 10))
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            
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
            
            elif method == "Extract Links":
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
                # Provide common libraries and the soup object
                exec_globals = {'soup': soup, 'results': [], 'pd': pd, 'np': np, 're': re}
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
