import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO, BytesIO
import requests
from bs4 import BeautifulSoup
import plotly.express as px

st.set_page_config(page_title="Data Analysis Suit", layout="wide")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def sql_query(data):
    st.subheader("SQL Query Tool")
    query = st.text_area("Write your SQL query here (e.g. SELECT * FROM data LIMIT 5):", height=150)
    if st.button("Run Query"):
        try:
            conn = sqlite3.connect(':memory:')
            data.to_sql('data', conn, index=False, if_exists='replace')
            result = pd.read_sql_query(query, conn)
            st.dataframe(result)
            conn.close()
        except Exception as e:
            st.error(f"SQL Error: {e}")

def eda_tools(data):
    st.subheader("Exploratory Data Analysis (EDA)")
    st.markdown("Select a tool below:")

    tool = st.selectbox("Choose EDA Tool", [
        "Summary Statistics",
        "Missing Values",
        "Value Counts",
        "Data Types",
        "Correlation Matrix",
        "Histogram",
        "Box Plot",
        "Scatter Plot",
        "Heatmap",
        "Pairplot"
    ])

    if tool == "Summary Statistics":
        st.write(data.describe())
    elif tool == "Missing Values":
        st.write(data.isnull().sum())
    elif tool == "Value Counts":
        col = st.selectbox("Select column for value counts", data.columns)
        st.write(data[col].value_counts())
    elif tool == "Data Types":
        st.write(data.dtypes)
    elif tool == "Correlation Matrix":
        corr = data.select_dtypes(include=np.number).corr()
        st.write(corr)
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax)
        st.pyplot(fig)
    elif tool == "Histogram":
        col = st.selectbox("Select numeric column for histogram", data.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        ax.hist(data[col].dropna())
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)
    elif tool == "Box Plot":
        col = st.selectbox("Select numeric column for box plot", data.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col])
        st.pyplot(fig)
    elif tool == "Scatter Plot":
        cols = st.multiselect("Select two numeric columns for scatter plot", data.select_dtypes(include=np.number).columns, default=data.select_dtypes(include=np.number).columns[:2])
        if len(cols) == 2:
            fig, ax = plt.subplots()
            ax.scatter(data[cols[0]], data[cols[1]])
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            st.pyplot(fig)
        else:
            st.info("Select exactly 2 columns")
    elif tool == "Heatmap":
        corr = data.select_dtypes(include=np.number).corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    elif tool == "Pairplot":
        sns_plot = sns.pairplot(data.select_dtypes(include=np.number))
        st.pyplot(sns_plot)

def excel_queries(data):
    st.subheader("Excel Query Tool")
    st.info("Upload Excel files with formulas or complex queries externally. Here you can preview and do basic formula-like operations.")
    st.write("Preview data:")
    st.dataframe(data.head())

def powerbi_dashboard(data):
    st.subheader("Power BI - Dashboard Generator")
    mode = st.radio("Select mode", ["Manual Dashboard", "Automatic Dashboard"])

    if mode == "Manual Dashboard":
        st.markdown("**Select columns and chart type to create custom visuals**")
        numeric_cols = list(data.select_dtypes(include=np.number).columns)
        cat_cols = list(data.select_dtypes(include=['object', 'category']).columns)
        col1 = st.selectbox("Select X axis column (categorical or numeric)", cat_cols + numeric_cols)
        col2 = st.selectbox("Select Y axis column (numeric)", numeric_cols)
        chart_type = st.selectbox("Select chart type", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"])

        if st.button("Generate Chart"):
            if chart_type == "Bar Chart":
                fig = px.bar(data, x=col1, y=col2)
            elif chart_type == "Line Chart":
                fig = px.line(data, x=col1, y=col2)
            elif chart_type == "Scatter Plot":
                fig = px.scatter(data, x=col1, y=col2)
            elif chart_type == "Pie Chart":
                if col1 in cat_cols:
                    fig = px.pie(data, names=col1, values=col2)
                else:
                    st.error("Pie Chart requires categorical column on X axis")
                    return
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("**Automatically generated dashboard with common charts**")
        numeric_cols = list(data.select_dtypes(include=np.number).columns)
        cat_cols = list(data.select_dtypes(include=['object', 'category']).columns)

        st.markdown("### Histograms")
        for col in numeric_cols[:3]:
            fig = px.histogram(data, x=col, nbins=30, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Boxplots")
        for col in numeric_cols[:3]:
            fig = px.box(data, y=col, title=f"Boxplot of {col}")
            st.plotly_chart(fig, use_container_width=True)

        if cat_cols:
            st.markdown("### Bar charts for categorical columns")
            for col in cat_cols[:3]:
                counts = data[col].value_counts().reset_index()
                counts.columns = [col, 'count']
                fig = px.bar(counts, x=col, y='count', title=f"Value counts of {col}")
                st.plotly_chart(fig, use_container_width=True)

def python_console(data):
    st.subheader("Python Advanced Console")
    st.info("Write Python code to analyze your dataframe named `df`.")
    code = st.text_area("Enter Python code here:", height=200, placeholder="e.g., df['new_col'] = df['col1'] + 10\nprint(df.head())")
    if st.button("Run Code"):
        local_vars = {'df': data.copy(), 'pd': pd, 'np': np}
        try:
            exec(code, {}, local_vars)
            if 'df' in local_vars:
                st.write(local_vars['df'].head())
        except Exception as e:
            st.error(f"Error running code: {e}")

def pandas_console(data):
    st.subheader("Pandas Advanced Console")
    st.info("Write Pandas commands to analyze your dataframe named `df`.")
    cmd = st.text_area("Enter Pandas command here:", height=150, placeholder="e.g., df.groupby('col1').mean()")
    if st.button("Run Pandas Command"):
        local_vars = {'df': data.copy(), 'pd': pd, 'np': np}
        try:
            result = eval(cmd, {}, local_vars)
            st.write(result)
        except Exception as e:
            st.error(f"Error running command: {e}")

def web_scraping():
    st.subheader("Web Scraping Tool")
    url = st.text_input("Enter URL to scrape")
    tag = st.text_input("Enter HTML tag to extract (optional)")
    attr_name = st.text_input("Enter attribute name to filter (optional)")
    attr_value = st.text_input("Enter attribute value to filter (optional)")
    if st.button("Scrape"):
        if not url:
            st.error("Please enter a URL")
            return
        try:
            res = requests.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            if tag:
                if attr_name and attr_value:
                    elements = soup.find_all(tag, {attr_name: attr_value})
                else:
                    elements = soup.find_all(tag)
                texts = [el.get_text(strip=True) for el in elements]
            else:
                texts = [soup.get_text(strip=True)]
            st.write(texts[:20])  # show top 20 results
        except Exception as e:
            st.error(f"Scraping error: {e}")

def main():
    st.title("Data Analysis Suit - Final Year Project")
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON supported)", type=['csv', 'xls', 'xlsx', 'json'])

    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
            st.success("Data loaded successfully!")
            st.sidebar.title("Tools")
            tools = [
                "SQL Query",
                "EDA Tools",
                "Excel Query",
                "Power BI Dashboard",
                "Python Console",
                "Pandas Console",
                "Web Scraping"
            ]
            tool_choice = st.sidebar.selectbox("Choose a tool", tools)

            if tool_choice == "SQL Query":
                sql_query(data)
            elif tool_choice == "EDA Tools":
                eda_tools(data)
            elif tool_choice == "Excel Query":
                excel_queries(data)
            elif tool_choice == "Power BI Dashboard":
                powerbi_dashboard(data)
            elif tool_choice == "Python Console":
                python_console(data)
            elif tool_choice == "Pandas Console":
                pandas_console(data)
            elif tool_choice == "Web Scraping":
                web_scraping()
        else:
            st.error("Failed to load data.")
    else:
        st.info("Please upload a dataset to get started.")

if __name__ == "__main__":
    main()
