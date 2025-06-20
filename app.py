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

def should_create_visualization(user_query: str) -> bool:
    """Check if user explicitly asks for visualization"""
    visualization_keywords = [
        'visualisasi', 'chart', 'graph', 'grafik', 'plot', 'diagram',
        'gambar', 'tampilkan grafik', 'buatkan chart', 'visualkan',
        'pie chart', 'bar chart', 'line chart', 'scatter', 'histogram',
        'heatmap', 'tren grafik', 'plot data', 'dashboard'
    ]
    
    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in visualization_keywords)

def create_advanced_visualization(df, user_query):
    """Create advanced visualization based on data patterns and user query"""
    if df.empty:
        return None, "No data available for visualization"
    
    # Convert columns to appropriate types
    df = df.copy()
    
    # Identify column types
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    
    for col in df.columns:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
            
        # Check for date columns
        if any(word in col.lower() for word in ['date', 'time', 'created', 'updated', 'tanggal']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isna().all():
                    date_cols.append(col)
                    continue
            except:
                pass
        
        # Check for numeric columns
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
    
    query_lower = user_query.lower()
    
    # Determine chart type based on query intent
    chart_type = "bar"  # default
    
    if any(word in query_lower for word in ['trend', 'waktu', 'time', 'bulan', 'tahun', 'seiring', 'berkembang', 'naik', 'turun', 'line']):
        chart_type = "line"
    elif any(word in query_lower for word in ['distribusi', 'pie', 'bagian', 'proporsi', 'persentase', 'pembagian']):
        chart_type = "pie"
    elif any(word in query_lower for word in ['scatter', 'hubungan', 'korelasi', 'vs', 'versus', 'pengaruh']):
        chart_type = "scatter"
    elif any(word in query_lower for word in ['heatmap', 'panas', 'matrix', 'matriks']):
        chart_type = "heatmap"
    elif any(word in query_lower for word in ['histogram', 'sebaran', 'frekuensi', 'distribusi nilai']):
        chart_type = "histogram"
    
    try:
        # Create visualization based on data structure and intent
        if chart_type == "line" and (date_cols or len(df) > 5) and numeric_cols:
            if date_cols:
                x_col = date_cols[0]
                df = df.sort_values(x_col)
            else:
                x_col = categorical_cols[0] if categorical_cols else df.columns[0]
            
            y_col = numeric_cols[0]
            
            fig = px.line(df, x=x_col, y=y_col,
                         title=f"Trend {y_col} over {x_col}",
                         markers=True)
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, height=500)
            return fig, f"Line chart showing trend of {y_col} over {x_col}"
        
        elif chart_type == "pie" and categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            df_agg = df.groupby(cat_col)[num_col].sum().reset_index()
            
            if len(df_agg) > 10:
                df_agg = df_agg.nlargest(10, num_col)
            
            fig = px.pie(df_agg, names=cat_col, values=num_col,
                        title=f"Distribution of {num_col} by {cat_col}")
            fig.update_layout(height=500)
            return fig, f"Pie chart showing distribution of {num_col} by {cat_col}"
        
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            color_col = categorical_cols[0] if categorical_cols else None
            
            fig = px.scatter(df, x=x_col, y=y_col,
                           color=color_col,
                           title=f"Relationship between {x_col} and {y_col}",
                           hover_data=df.columns.tolist())
            fig.update_layout(height=500)
            return fig, f"Scatter plot showing relationship between {x_col} and {y_col}"
        
        elif chart_type == "histogram" and numeric_cols:
            num_col = numeric_cols[0]
            fig = px.histogram(df, x=num_col, nbins=min(20, len(df.nunique())),
                             title=f"Distribution of {num_col}")
            fig.update_layout(height=500)
            return fig, f"Histogram showing distribution of {num_col}"
        
        elif chart_type == "heatmap" and len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                           title="Correlation Matrix",
                           color_continuous_scale="RdBu",
                           aspect="auto",
                           text_auto=True)
            fig.update_layout(height=500)
            return fig, "Correlation heatmap of numeric variables"
        
        elif categorical_cols and numeric_cols:
            # Default: Bar chart
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
            
            fig = px.bar(df_agg, 
                        x=cat_col, 
                        y=num_col,
                        title=f"{num_col} by {cat_col}{title_suffix}",
                        color=num_col,
                        color_continuous_scale="viridis")
            
            fig.update_xaxis(tickangle=45)
            fig.update_layout(height=500, showlegend=False)
            
            return fig, f"Bar chart showing {num_col} by {cat_col}{title_suffix.lower()}"
        
        elif len(df.columns) >= 2:
            col1, col2 = df.columns[0], df.columns[1]
            
            if pd.api.types.is_numeric_dtype(df[col2]):
                df_plot = df.head(20)
                fig = px.bar(df_plot, x=col1, y=col2,
                           title=f"{col2} by {col1}")
                fig.update_xaxis(tickangle=45)
                fig.update_layout(height=500)
                return fig, f"Bar chart of {col2} by {col1}"
            else:
                df_counts = df[col1].value_counts().head(20).reset_index()
                df_counts.columns = [col1, 'Count']
                
                fig = px.bar(df_counts, x=col1, y='Count',
                           title=f"Count of {col1}")
                fig.update_xaxis(tickangle=45)
                fig.update_layout(height=500)
                return fig, f"Count chart of {col1}"
        
        else:
            col = df.columns[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                fig.update_layout(height=500)
                return fig, f"Histogram of {col}"
            else:
                df_counts = df[col].value_counts().head(20).reset_index()
                df_counts.columns = [col, 'Count']
                
                fig = px.bar(df_counts, x=col, y='Count',
                           title=f"Count of {col}")
                fig.update_xaxis(tickangle=45)
                fig.update_layout(height=500)
                return fig, f"Count chart of {col}"
    
    except Exception as e:
        return None, f"Error creating chart: {str(e)}"
    
    return None, "Unable to create appropriate visualization for this data"

def execute_sql_for_dataframe(query: str, db: SQLDatabase) -> pd.DataFrame:
    """Execute SQL and return clean DataFrame"""
    try:
        cleaned_query = clean_sql_output(query)
        
        if not is_valid_select(cleaned_query):
            return pd.DataFrame()
        
        try:
            # Execute query using SQLDatabase's method
            result = db._execute(cleaned_query)
            
            # Get column names and data
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
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            # Convert data types appropriately
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
            # Fallback to string-based approach
            result = db.run(cleaned_query)
            return parse_string_result_to_dataframe(result)
    
    except Exception as e:
        return pd.DataFrame()

def parse_string_result_to_dataframe(result: str) -> pd.DataFrame:
    """Fallback function to parse string results to DataFrame"""
    try:
        if not result or not result.strip():
            return pd.DataFrame()
        
        lines = result.strip().split('\n')
        if len(lines) < 1:
            return pd.DataFrame()
        
        data_rows = []
        headers = []
        
        # Handle different formats
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
        
        # Try tab-separated format
        if '\t' in lines[0]:
            headers = [h.strip() for h in lines[0].split('\t') if h.strip()]
            for line in lines[1:]:
                if line.strip():
                    row = [cell.strip() if cell.strip() else None for cell in line.split('\t')]
                    while len(row) < len(headers):
                        row.append(None)
                    data_rows.append(row[:len(headers)])
        else:
            # Try comma-separated format
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
    You are a MySQL expert working with a comprehensive business database. Generate efficient SQL queries that directly answer the user's question with detailed, complete results.
    
    Database contains these tables with full access:
    - khs_customer_transactions: Complete customer transaction records (32 columns)
    - returns: Customer return records 
    - sales_targets: Monthly sales targets by product
    - produk: Product master data
    - iventory: Current inventory levels
    - All other tables in the database are accessible
    
    Query Guidelines:
    - Write comprehensive SQL queries that provide complete answers
    - Use appropriate JOINs to get related data from multiple tables
    - Include relevant columns that provide context and detail
    - Use proper aggregation functions (SUM, COUNT, AVG, etc.) when needed
    - Apply appropriate filters and sorting
    - Use LIMIT only when dealing with very large datasets (>1000 rows)
    - Include date ranges, grouping, and ordering as appropriate
    - Access all available tables and columns as needed
    - Write only the SQL query without formatting, backticks, or explanations
    
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
    """Generate comprehensive natural language response"""
    sql_chain = get_sql_chain(db)

    template = """
    Anda adalah seorang analis bisnis yang berpengalaman dan ahli dalam menganalisis data perusahaan. Berikan jawaban yang comprehensive, detail, dan insightful dalam bahasa Indonesia berdasarkan hasil SQL.
    
    Panduan untuk memberikan jawaban yang berkualitas:
    1. Berikan analisis yang mendalam dan detail, tidak hanya angka mentah
    2. Jelaskan konteks bisnis dari data yang ditemukan
    3. Identifikasi tren, pola, atau insight penting
    4. Berikan interpretasi yang meaningful dari hasil data
    5. Sertakan implikasi bisnis dan rekomendasi jika relevan
    6. Gunakan format yang mudah dibaca dengan struktur yang jelas
    7. Bandingkan data dengan konteks bisnis yang lebih luas jika memungkinkan
    8. Berikan detail yang cukup untuk membantu pengambilan keputusan
    
    <SCHEMA>{schema}</SCHEMA>
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    
    Berikan analisis bisnis yang komprehensif dan detail berdasarkan data:
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(
        model="deepseek/deepseek-chat:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.1
    )

    def execute_cleaned_sql(vars):
        raw_query = vars["query"]
        cleaned_query = clean_sql_output(raw_query)
        
        if not is_valid_select(cleaned_query):
            raise ValueError("Only SELECT queries are allowed for security reasons.")
        
        try:
            result = db.run(cleaned_query)
            return result
        except Exception as e:
            return f"SQL Execution Error: {str(e)}"

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
                    "Analisis performa penjualan bulan ini",
                    "Siapa 10 customer dengan transaksi terbesar?",
                    "Bagaimana tren penjualan 6 bulan terakhir?",
                    "Produk apa yang sering dikembalikan?",
                    "Perbandingan target vs actual sales",
                    "Distribusi penjualan per provinsi",
                    "Visualisasi trend penjualan bulanan",
                    "Buatkan chart top products"
                ]
                
                for sample in samples:
                    st.markdown(f"• {sample}")

# ============= STREAMLIT APP =============
st.set_page_config(
    page_title="BI Chat - Advanced Business Intelligence", 
    page_icon="📈", 
    layout="wide"
)

st.title("🏢 Business Intelligence Chat - Advanced Analytics & Visualization")
st.markdown("Tanyakan apa saja tentang data bisnis Anda dan dapatkan insight mendalam! Tambahkan kata 'visualisasi' atau 'chart' untuk grafik.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Halo! Saya adalah asisten Business Intelligence Anda. Saya dapat menganalisis data bisnis Anda secara mendalam dan memberikan insight yang komprehensif. Untuk visualisasi, tambahkan kata 'visualisasi', 'chart', atau 'grafik' dalam pertanyaan Anda!")
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

    # Connection status
    if st.session_state.db:
        st.success("🟢 Database Connected")
    else:
        st.error("🔴 Database Not Connected")

# Display table information
if st.session_state.db:
    display_table_info(st.session_state.db)

# Main interface
if st.session_state.db:
    # Chat interface
    st.subheader("💬 Chat with Your Data")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
                st.markdown(message.content)
    
    # Chat input
    if user_query := st.chat_input("Tanyakan tentang data bisnis Anda..."):
        # Add user message
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Menganalisis data Anda..."):
                try:
                    # Generate and execute SQL query
                    sql_chain = get_sql_chain(st.session_state.db)
                    sql_query = sql_chain.invoke({
                        "question": user_query,
                        "schema": st.session_state.db.get_table_info()
                    })
                    
                    # Get comprehensive response
                    response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Check if visualization is requested
                    if should_create_visualization(user_query):
                        # Create and display visualization
                        df = execute_sql_for_dataframe(sql_query, st.session_state.db)
                        
                        if not df.empty:
                            st.subheader("📊 Data Visualization")
                            
                            # Show data preview
                            with st.expander("📋 View Raw Data"):
                                st.dataframe(df, use_container_width=True)
                                st.caption(f"Showing {len(df)} rows × {len(df.columns)} columns")
                            
                            # Create visualization
                            fig, chart_description = create_advanced_visualization(df, user_query)
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption(f"📈 {chart_description}")
                                
                                # Additional insights
                                st.markdown("### 📋 Quick Data Summary")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Records", len(df))
                                with col2:
                                    st.metric("Columns", len(df.columns))
                                with col3:
                                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                                    if len(numeric_cols) > 0:
                                        st.metric("Numeric Columns", len(numeric_cols))
                            else:
                                st.info("📊 Data tersedia namun tidak dapat membuat visualisasi yang sesuai untuk dataset ini.")
                        else:
                            st.info("📊 Query tidak mengembalikan data. Silakan coba pertanyaan lain.")
                    
                except Exception as e:
                    error_msg = f"❌ Error memproses pertanyaan Anda: {str(e)}"
                    st.error(error_msg)
                    response = error_msg
        
        # Add assistant response to history
        st.session_state.chat_history.append(AIMessage(content=response))

else:
    # Not connected state
    st.warning("⚠️ Silakan sambungkan ke database terlebih dahulu menggunakan sidebar.")
    st.info("Setelah terhubung, Anda dapat bertanya seperti:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Analisis Penjualan:**
        - Analisis mendalam performa penjualan bulan ini
        - Bagaimana tren revenue 6 bulan terakhir?
        - Siapa customer dengan kontribusi terbesar?
        """)
    
    with col2:
        st.markdown("""
        **Insights & Visualisasi:**
        - Visualisasi perbandingan actual vs target
        - Buatkan chart distribusi penjualan per region
        - Grafik trend pertumbuhan customer baru
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚀 <strong>BI Chat</strong> - Advanced Business Intelligence with Smart Analytics</p>
        <p>💡 <em>Dapatkan insight mendalam dengan analisis komprehensif dan visualisasi cerdas</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)
