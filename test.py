from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
import numpy as np
from datetime import datetime, timedelta

load_dotenv()

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def clean_sql_output(query: str) -> str:
    """Clean SQL query from prefixes and formatting"""
    query = query.strip()
    if query.startswith("```sql"):
        query = query[6:]
    if query.endswith("```"):
        query = query[:-3]
    if query.startswith("`") and query.endswith("`"):
        query = query[1:-1]
    if query.lower().startswith("sql "):
        query = query[4:]
    query = query.replace("`", "")
    return query.strip()

def is_valid_select(query: str) -> bool:
    """Validate that query is a SELECT statement"""
    cleaned_query = clean_sql_output(query)
    return cleaned_query.strip().lower().startswith("select")

def create_visualization(df, user_query):
    """Create visualization only when explicitly requested"""
    if df.empty:
        return None, ""
    
    # Convert columns to appropriate types
    df = df.copy()
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        if any(word in col.lower() for word in ['date', 'time', 'created', 'updated', 'tanggal']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isna().all():
                    date_cols.append(col)
                    continue
            except:
                pass
        
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)
        else:
            try:
                numeric_series = pd.to_numeric(col_data, errors='coerce')
                non_null_numeric = numeric_series.dropna()
                if len(non_null_numeric) > len(col_data) * 0.7:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except:
                categorical_cols.append(col)
    
    try:
        if date_cols and numeric_cols:
            x_col = date_cols[0]
            df = df.sort_values(x_col)
            y_col = numeric_cols[0]
            fig = px.line(df, x=x_col, y=y_col, title=f"Trend {y_col} over {x_col}", markers=True)
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, height=500)
            return fig, f"Line chart showing trend of {y_col} over {x_col}"
        
        elif categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            if df[cat_col].duplicated().any():
                df_agg = df.groupby(cat_col)[num_col].sum().reset_index()
            else:
                df_agg = df[[cat_col, num_col]].copy()
            
            df_agg = df_agg.sort_values(num_col, ascending=False)
            max_items = 20
            if len(df_agg) > max_items:
                df_agg = df_agg.head(max_items)
                title_suffix = f" (Top {max_items})"
            else:
                title_suffix = ""
            
            fig = px.bar(df_agg, x=cat_col, y=num_col,
                        title=f"{num_col} by {cat_col}{title_suffix}",
                        color=num_col,
                        color_continuous_scale="viridis")
            
            fig.update_xaxis(tickangle=45)
            fig.update_layout(height=500, showlegend=False)
            return fig, f"Bar chart showing {num_col} by {cat_col}{title_suffix.lower()}"
        
        elif len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Relationship between {x_col} and {y_col}")
            fig.update_layout(height=500)
            return fig, f"Scatter plot showing relationship between {x_col} and {y_col}"
        
        elif numeric_cols:
            num_col = numeric_cols[0]
            fig = px.histogram(df, x=num_col, title=f"Distribution of {num_col}")
            fig.update_layout(height=500)
            return fig, f"Histogram showing distribution of {num_col}"
        
        elif len(df.columns) >= 2:
            col1, col2 = df.columns[0], df.columns[1]
            df_counts = df[col1].value_counts().head(20).reset_index()
            df_counts.columns = [col1, 'Count']
            fig = px.bar(df_counts, x=col1, y='Count', title=f"Count of {col1}")
            fig.update_xaxis(tickangle=45)
            fig.update_layout(height=500)
            return fig, f"Count chart of {col1}"
        
    except Exception as e:
        return None, ""

def execute_sql_for_dataframe(query: str, db: SQLDatabase) -> pd.DataFrame:
    """Execute SQL and return DataFrame"""
    try:
        cleaned_query = clean_sql_output(query)
        
        if not is_valid_select(cleaned_query):
            return pd.DataFrame()
        
        try:
            result = db._execute(cleaned_query)
            
            if hasattr(result, 'description') and result.description:
                columns = [desc[0] for desc in result.description]
                rows = result.fetchall()
            else:
                import sqlalchemy
                with db._engine.connect() as conn:
                    result = conn.execute(sqlalchemy.text(cleaned_query))
                    columns = list(result.keys())
                    rows = result.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=columns)
            
            for col in df.columns:
                df[col] = df[col].replace([None, 'NULL', 'null', ''], pd.NA)
                
                if df[col].dtype == 'object':
                    try:
                        non_null_values = df[col].dropna()
                        if len(non_null_values) > 0:
                            sample_val = str(non_null_values.iloc[0])
                            if sample_val.replace('.', '').replace('-', '').replace('+', '').isdigit():
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
                    
                    if any(word in col.lower() for word in ['date', 'time', 'created', 'updated']):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
            
            return df
            
        except Exception as e:
            result = db.run(cleaned_query)
            return parse_string_result_to_dataframe(result)
    
    except Exception as e:
        return pd.DataFrame()

def parse_string_result_to_dataframe(result: str) -> pd.DataFrame:
    """Parse string results to DataFrame"""
    try:
        if not result or not result.strip():
            return pd.DataFrame()
        
        lines = result.strip().split('\n')
        if len(lines) < 1:
            return pd.DataFrame()
        
        data_rows = []
        headers = []
        
        if '[' in result and ']' in result:
            try:
                import ast
                parsed_data = ast.literal_eval(result.strip())
                if isinstance(parsed_data, (list, tuple)) and len(parsed_data) > 0:
                    first_row = parsed_data[0]
                    if isinstance(first_row, (list, tuple)):
                        headers = [f"col_{i}" for i in range(len(first_row))]
                        data_rows = [list(row) for row in parsed_data]
                    else:
                        headers = ['value']
                        data_rows = [[row] for row in parsed_data]
                    return pd.DataFrame(data_rows, columns=headers)
            except:
                pass
        
        if '\t' in lines[0]:
            headers = [h.strip() for h in lines[0].split('\t') if h.strip()]
            for line in lines[1:]:
                if line.strip():
                    row = [cell.strip() if cell.strip() else None for cell in line.split('\t')]
                    while len(row) < len(headers):
                        row.append(None)
                    data_rows.append(row[:len(headers)])
        else:
            headers = [h.strip() for h in lines[0].split(',') if h.strip()]
            for line in lines[1:]:
                if line.strip():
                    row = [cell.strip() if cell.strip() else None for cell in line.split(',')]
                    while len(row) < len(headers):
                        row.append(None)
                    data_rows.append(row[:len(headers)])
        
        if not data_rows or not headers:
            headers = ['data']
            data_rows = [[line.strip()] for line in lines if line.strip()]
        
        return pd.DataFrame(data_rows, columns=headers)
        
    except Exception as e:
        return pd.DataFrame()

def get_sql_chain(db):
    """Create SQL generation chain"""
    prompt_template = """
    You are a MySQL expert working with a business database. Generate efficient SQL queries that directly answer the user's question.
    
    Available tables:
    1. returns - Customer return records (RETURN_ID, ITEM_ID, RETURN_DATE, RETURNED_QUANTITY, REASON)
    2. sales_targets - Monthly sales targets (ITEM_ID, YEAR, MONTH, TARGET_QUANTITY, TARGET_REVENUE)
    3. produk - Product master data (ITEM_ID, ITEM_CODE, ITEM_DESCRIPTION, ITEM_TYPE)
    4. iventory - Current stock levels (ITEM_ID, STOCK_QUANTITY, LAST_UPDATED)
    5. khs_customer_transactions - Detailed customer transactions (32 columns including customer info, product details, pricing, location, coordinates)
    
    Important notes:
    - Use correct table names: sales_targets, iventory, khs_customer_transactions
    - Write only the SQL query without formatting or backticks
    - Focus on providing relevant data for visualization
    - Always use LIMIT for large datasets (default LIMIT 100 unless specifically asked for more)
    - Use proper JOINs when needed
    - For date-based queries, use appropriate date functions
    - Group data when showing aggregated results
    
    <SCHEMA>{schema}</SCHEMA>
    
    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    llm = ChatOpenAI(
        model="meta-llama/llama-3.3-8b-instruct:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0
    )

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    """Generate natural language response"""
    sql_chain = get_sql_chain(db)

    template = """
    Anda adalah seorang analis data bisnis yang berpengalaman. Berikan jawaban yang mendalam dan komprehensif dalam Bahasa Indonesia berdasarkan hasil SQL dan konteks pertanyaan.
    
    Pedoman:
    1. Jawab dengan detail dan analisis mendalam
    2. Gunakan angka aktual dari respons SQL
    3. Sertakan wawasan bisnis yang relevan
    4. Gunakan Bahasa Indonesia yang natural dan profesional
    5. Berikan konteks tambahan yang bermanfaat
    6. Jika diminta visualisasi, jelaskan data yang akan divisualisasikan
    
    <SCHEMA>{schema}</SCHEMA>
    SQL Query: <SQL>{query}</SQL>
    Pertanyaan pengguna: {question}
    SQL Response: {response}
    
    Berikan jawaban analitis yang mendalam:
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(
        model="deepseek/deepseek-chat:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.3
    )

    def execute_cleaned_sql(vars):
        raw_query = vars["query"]
        cleaned_query = clean_sql_output(raw_query)
        
        if not is_valid_select(cleaned_query):
            raise ValueError("Hanya query SELECT yang diperbolehkan.")
        
        try:
            result = db.run(cleaned_query)
            return result
        except Exception as e:
            return f"Error SQL: {str(e)}"

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=execute_cleaned_sql,
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

def display_table_info(db):
    """Display database table information"""
    if db:
        with st.expander("📊 Database Schema & Sample Queries"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Available Tables:")
                table_info = {
                    "**khs_customer_transactions**": "Complete transaction data (32 columns)",
                    "**returns**": "Product return records", 
                    "**sales_targets**": "Monthly sales targets",
                    "**produk**": "Product master data",
                    "**iventory**": "Current stock levels"
                }
                
                for table, desc in table_info.items():
                    st.markdown(f"{table}: {desc}")
            
            with col2:
                st.subheader("💡 Sample Questions:")
                samples = [
                    "Visualisasi trend penjualan 6 bulan terakhir",
                    "Apa saja produk terlaris bulan ini?",
                    "Total pendapatan per provinsi",
                    "Produk apa yang paling sering direturn?",
                    "Visualisasi perbandingan target vs aktual penjualan",
                    "Distribusi geografis penjualan",
                    "Pola transaksi pelanggan"
                ]
                
                for sample in samples:
                    st.markdown(f"• {sample}")

# ============= STREAMLIT APP =============
st.set_page_config(
    page_title="BI Chat with Advanced Visualization", 
    page_icon="📈", 
    layout="wide"
)

st.title("🏢 Business Intelligence Chat with Advanced Data Visualization")
st.markdown("Ask questions about your business data and get instant insights with smart visualizations!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Halo! Saya asisten Business Intelligence Anda. Saya dapat membantu menganalisis data bisnis Anda dan membuat visualisasi ketika diminta. Apa yang ingin Anda ketahui?")
    ]

if "db" not in st.session_state:
    st.session_state.db = None

# Sidebar for database connection
with st.sidebar:
    st.subheader("🔗 Database Connection")
    
    with st.form("db_connection"):
        host = st.text_input("Host", value="localhost")
        port = st.text_input("Port", value="3306")
        user = st.text_input("User", value="root")
        password = st.text_input("Password", type="password", value="admin")
        database = st.text_input("Database", value="business_db")
        
        connect_btn = st.form_submit_button("🚀 Connect to Database")
        
        if connect_btn:
            with st.spinner("Connecting to database..."):
                try:
                    db = init_database(user, password, host, port, database)
                    st.session_state.db = db
                    st.success("✅ Successfully connected!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")

    if st.session_state.db:
        st.success("🟢 Database Connected")
    else:
        st.error("🔴 Database Not Connected")

# Display table information
if st.session_state.db:
    display_table_info(st.session_state.db)

# Main interface
if st.session_state.db:
    st.subheader("💬 Chat with Your Data")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
                st.markdown(message.content)
    
    if user_query := st.chat_input("Ask about your business data..."):
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                try:
                    # Generate SQL query
                    sql_chain = get_sql_chain(st.session_state.db)
                    sql_query = sql_chain.invoke({
                        "question": user_query,
                        "schema": st.session_state.db.get_table_info()
                    })
                    
                    # Get response
                    response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Only create visualization if explicitly requested
                    if "visualisasi" in user_query.lower() or "visualization" in user_query.lower():
                        df = execute_sql_for_dataframe(sql_query, st.session_state.db)
                        
                        if not df.empty:
                            st.subheader("📊 Data Visualization")
                            
                            with st.expander("📋 View Raw Data"):
                                st.dataframe(df, use_container_width=True)
                            
                            fig, chart_description = create_visualization(df, user_query)
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption(f"📈 {chart_description}")
                            else:
                                st.info("Visualisasi tidak dapat dibuat untuk data ini.")
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    response = error_msg
        
        st.session_state.chat_history.append(AIMessage(content=response))

else:
    st.warning("⚠️ Please connect to your database first using the sidebar.")
    st.info("Contoh pertanyaan yang bisa diajukan:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Analisis Penjualan:**
        - Produk terlaris bulan ini
        - Trend penjualan tahun ini
        - Pelanggan dengan transaksi terbanyak
        """)
    
    with col2:
        st.markdown("""
        **Visualisasi Data:**
        - Visualisasi trend penjualan 6 bulan terakhir
        - Visualisasi perbandingan target vs aktual
        - Visualisasi distribusi geografis penjualan
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚀 <strong>BI Chat</strong> - Powered by AI for Smart Business Intelligence</p>
    </div>
    """, 
    unsafe_allow_html=True
)
