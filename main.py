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
    # Remove backticks
    if query.startswith("```sql"):
        query = query[6:]
    if query.endswith("```"):
        query = query[:-3]
    if query.startswith("`") and query.endswith("`"):
        query = query[1:-1]
    # Remove sql prefix (case insensitive)
    if query.lower().startswith("sql "):
        query = query[4:]
    # Remove any remaining backticks
    query = query.replace("`", "")
    return query.strip()

def is_valid_select(query: str) -> bool:
    """Validate that query is a SELECT statement"""
    cleaned_query = clean_sql_output(query)
    return cleaned_query.strip().lower().startswith("select")

def create_advanced_visualization(df, user_query):
    """Create advanced visualization based on data patterns and user query"""
    if df.empty:
        return None, "No data available for visualization"
    
    st.write(f"Debug: DataFrame shape: {df.shape}")
    st.write(f"Debug: DataFrame columns: {list(df.columns)}")
    st.write("Debug: DataFrame head:")
    st.dataframe(df.head(), use_container_width=True)
    
    # Convert columns to appropriate types
    df = df.copy()
    
    # Identify column types with better detection
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    
    for col in df.columns:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
            
        # Check for date columns first
        if any(word in col.lower() for word in ['date', 'time', 'created', 'updated', 'tanggal']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isna().all():
                    date_cols.append(col)
                    continue
            except:
                pass
        
        # Check for numeric columns with better logic
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)
        else:
            # Try to convert to numeric
            try:
                numeric_series = pd.to_numeric(col_data, errors='coerce')
                non_null_numeric = numeric_series.dropna()
                
                # If most values can be converted to numeric, treat as numeric
                if len(non_null_numeric) > len(col_data) * 0.7:  # 70% threshold
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except:
                categorical_cols.append(col)
    
    st.write(f"Debug: Numeric columns: {numeric_cols}")
    st.write(f"Debug: Categorical columns: {categorical_cols}")
    st.write(f"Debug: Date columns: {date_cols}")
    
    query_lower = user_query.lower()
    
    # Determine chart type based on query intent and data structure
    chart_type = "bar"  # default
    
    # Query intent analysis with Indonesian keywords
    if any(word in query_lower for word in ['trend', 'waktu', 'time', 'bulan', 'tahun', 'seiring', 'berkembang', 'naik', 'turun']):
        chart_type = "line"
    elif any(word in query_lower for word in ['distribusi', 'pie', 'bagian', 'proporsi', 'persentase', 'pembagian']):
        chart_type = "pie"
    elif any(word in query_lower for word in ['scatter', 'hubungan', 'korelasi', 'vs', 'versus', 'pengaruh']):
        chart_type = "scatter"
    elif any(word in query_lower for word in ['heatmap', 'panas', 'matrix', 'matriks']):
        chart_type = "heatmap"
    elif any(word in query_lower for word in ['histogram', 'sebaran', 'frekuensi', 'distribusi nilai']):
        chart_type = "histogram"
    elif any(word in query_lower for word in ['top', 'tertinggi', 'terbesar', 'ranking', 'urutan', 'terbaik']):
        chart_type = "bar"
    
    try:
        # Create visualization based on data structure and intent
        if chart_type == "line" and (date_cols or len(df) > 5) and numeric_cols:
            # Time series or sequential line chart
            if date_cols:
                x_col = date_cols[0]
                # Sort by date
                df = df.sort_values(x_col)
            else:
                # Use index or first categorical column as x-axis
                x_col = categorical_cols[0] if categorical_cols else df.columns[0]
            
            y_col = numeric_cols[0]
            
            fig = px.line(df, x=x_col, y=y_col,
                         title=f"Trend {y_col} over {x_col}",
                         markers=True)
            fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, height=500)
            return fig, f"Line chart showing trend of {y_col} over {x_col}"
        
        elif chart_type == "pie" and categorical_cols and numeric_cols:
            # Pie chart for categorical distribution
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Aggregate data if needed
            df_agg = df.groupby(cat_col)[num_col].sum().reset_index()
            
            # Limit to top categories for readability
            if len(df_agg) > 10:
                df_agg = df_agg.nlargest(10, num_col)
            
            fig = px.pie(df_agg, names=cat_col, values=num_col,
                        title=f"Distribution of {num_col} by {cat_col}")
            fig.update_layout(height=500)
            return fig, f"Pie chart showing distribution of {num_col} by {cat_col}"
        
        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            # Scatter plot for correlation
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
            # Histogram for distribution
            num_col = numeric_cols[0]
            fig = px.histogram(df, x=num_col, nbins=min(20, len(df.nunique())),
                             title=f"Distribution of {num_col}")
            fig.update_layout(height=500)
            return fig, f"Histogram showing distribution of {num_col}"
        
        elif chart_type == "heatmap" and len(numeric_cols) >= 2:
            # Correlation heatmap for multiple numeric columns
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                           title="Correlation Matrix",
                           color_continuous_scale="RdBu",
                           aspect="auto",
                           text_auto=True)
            fig.update_layout(height=500)
            return fig, "Correlation heatmap of numeric variables"
        
        elif categorical_cols and numeric_cols:
            # Default: Bar chart (most common case)
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Aggregate data if there are duplicates
            if df[cat_col].duplicated().any():
                df_agg = df.groupby(cat_col)[num_col].sum().reset_index()
            else:
                df_agg = df[[cat_col, num_col]].copy()
            
            # Sort by numeric value for better visualization
            df_agg = df_agg.sort_values(num_col, ascending=False)
            
            # Limit to top results for readability
            max_items = 20
            if len(df_agg) > max_items:
                df_agg = df_agg.head(max_items)
                title_suffix = f" (Top {max_items})"
            else:
                title_suffix = ""
            
            # Create bar chart with color gradient
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
            # Fallback: Simple visualization of available data
            col1, col2 = df.columns[0], df.columns[1]
            
            # Try to create appropriate chart based on data types
            if pd.api.types.is_numeric_dtype(df[col2]):
                # Numeric y-axis: bar chart
                df_plot = df.head(20)  # Limit rows
                fig = px.bar(df_plot, x=col1, y=col2,
                           title=f"{col2} by {col1}")
                fig.update_xaxis(tickangle=45)
                fig.update_layout(height=500)
                return fig, f"Bar chart of {col2} by {col1}"
            else:
                # Both categorical: count plot
                df_counts = df[col1].value_counts().head(20).reset_index()
                df_counts.columns = [col1, 'Count']
                
                fig = px.bar(df_counts, x=col1, y='Count',
                           title=f"Count of {col1}")
                fig.update_xaxis(tickangle=45)
                fig.update_layout(height=500)
                return fig, f"Count chart of {col1}"
        
        else:
            # Single column: count or distribution
            col = df.columns[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric: histogram
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                fig.update_layout(height=500)
                return fig, f"Histogram of {col}"
            else:
                # Categorical: count
                df_counts = df[col].value_counts().head(20).reset_index()
                df_counts.columns = [col, 'Count']
                
                fig = px.bar(df_counts, x=col, y='Count',
                           title=f"Count of {col}")
                fig.update_xaxis(tickangle=45)
                fig.update_layout(height=500)
                return fig, f"Count chart of {col}"
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.error(f"DataFrame info: {df.dtypes}")
        
        # Create a simple fallback visualization
        try:
            if len(df.columns) >= 2:
                fig = px.scatter(df.head(100), x=df.columns[0], y=df.columns[1],
                               title="Data Scatter Plot")
                fig.update_layout(height=500)
                return fig, "Simple scatter plot of available data"
        except:
            pass
        
        return None, f"Error creating chart: {str(e)}"
    
    return None, "Unable to create appropriate visualization for this data"

def execute_sql_for_dataframe(query: str, db: SQLDatabase) -> pd.DataFrame:
    """Execute SQL and return clean DataFrame - FIXED VERSION"""
    try:
        cleaned_query = clean_sql_output(query)
        st.write(f"Debug: Cleaned SQL Query: {cleaned_query}")
        
        if not is_valid_select(cleaned_query):
            st.error("Only SELECT queries are allowed")
            return pd.DataFrame()
        
        # Execute query using SQLDatabase's method that returns structured data
        try:
            # Use the database connection directly to get structured results
            result = db._execute(cleaned_query)
            st.write(f"Debug: SQL execution successful, processing results...")
            
            # Get column names from the cursor description
            if hasattr(result, 'description') and result.description:
                columns = [desc[0] for desc in result.description]
                rows = result.fetchall()
            else:
                # Fallback: try different approach
                import sqlalchemy
                with db._engine.connect() as conn:
                    result = conn.execute(sqlalchemy.text(cleaned_query))
                    columns = list(result.keys())
                    rows = result.fetchall()
            
            st.write(f"Debug: Got {len(rows)} rows with columns: {columns}")
            
            if not rows:
                st.warning("Query returned no data")
                return pd.DataFrame()
            
            # Create DataFrame directly from structured data
            df = pd.DataFrame(rows, columns=columns)
            st.write(f"Debug: DataFrame created successfully with shape: {df.shape}")
            
            # Convert data types appropriately
            for col in df.columns:
                # Handle None/NULL values
                df[col] = df[col].replace([None, 'NULL', 'null', ''], pd.NA)
                
                # Try to infer and convert appropriate data types
                if df[col].dtype == 'object':
                    # Try numeric conversion
                    try:
                        # Check if the column looks numeric
                        non_null_values = df[col].dropna()
                        if len(non_null_values) > 0:
                            # Try to convert a sample
                            sample_val = str(non_null_values.iloc[0])
                            if sample_val.replace('.', '').replace('-', '').replace('+', '').isdigit():
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                    except:
                        pass
                    
                    # Try date conversion for date-like columns
                    if any(word in col.lower() for word in ['date', 'time', 'created', 'updated']):
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
            
            return df
            
        except Exception as e:
            st.error(f"Error executing SQL with structured approach: {str(e)}")
            
            # Fallback: try the original string-based approach
            result = db.run(cleaned_query)
            return parse_string_result_to_dataframe(result)
    
    except Exception as e:
        st.error(f"Error executing SQL: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def parse_string_result_to_dataframe(result: str) -> pd.DataFrame:
    """Fallback function to parse string results to DataFrame"""
    try:
        if not result or not result.strip():
            return pd.DataFrame()
        
        lines = result.strip().split('\n')
        if len(lines) < 1:
            return pd.DataFrame()
        
        # Handle different formats
        data_rows = []
        headers = []
        
        # Try to parse as structured data
        if '[' in result and ']' in result:
            # Handle list/tuple format: [(1, 'name'), (2, 'name2')]
            try:
                import ast
                parsed_data = ast.literal_eval(result.strip())
                if isinstance(parsed_data, (list, tuple)) and len(parsed_data) > 0:
                    # Generate column names
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
            # Last resort: treat as single column
            headers = ['data']
            data_rows = [[line.strip()] for line in lines if line.strip()]
        
        return pd.DataFrame(data_rows, columns=headers)
        
    except Exception as e:
        st.error(f"Error parsing string result: {str(e)}")
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
    You are a business data analyst. Provide a concise, specific answer in Indonesian based on the SQL results.
    
    Guidelines:
    1. Answer directly what the user asked
    2. Use actual numbers from the SQL response
    3. Keep it concise and focused
    4. Use natural Indonesian language
    5. Don't provide unnecessary analysis unless asked
    
    <SCHEMA>{schema}</SCHEMA>
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    
    Provide a specific answer based on the data:
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
                    "Show top 10 customers by total sales",
                    "What are the best selling products this month?",
                    "Total revenue by province",
                    "Which products are returned most frequently?",
                    "Sales trend over the last 6 months",
                    "Compare actual vs target sales",
                    "Geographic distribution of sales",
                    "Customer transaction patterns"
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
        AIMessage(content="Hello! I'm your Business Intelligence assistant with advanced visualization capabilities. Ask me anything about your sales, customers, products, or targets!")
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

# Main interface with improved layout
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
    if user_query := st.chat_input("Ask about your business data..."):
        # Add user message
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
                        else:
                            st.info("📊 Data retrieved but no suitable visualization could be created for this dataset.")
                        
                        # Additional insights
                        if len(df) > 0:
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
                        st.info("📊 No data returned from the query. Please try a different question or check your database.")
                    
                except Exception as e:
                    error_msg = f"❌ Error processing your request: {str(e)}"
                    st.error(error_msg)
                    response = error_msg
                    
                    # Show debug info for troubleshooting
                    with st.expander("🔍 Debug Information"):
                        st.text(f"Error details: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Add assistant response to history
        st.session_state.chat_history.append(AIMessage(content=response))

else:
    # Not connected state
    st.warning("⚠️ Please connect to your database first using the sidebar.")
    st.info("Once connected, you can ask questions like:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Sales Analysis:**
        - Show me top 10 products by sales
        - What's the revenue trend this year?
        - Which customers buy the most?
        """)
    
    with col2:
        st.markdown("""
        **Performance Insights:**
        - Compare actual vs target sales
        - Which regions perform best?
        - What products are returned most?
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚀 <strong>BI Chat</strong> - Powered by AI for Smart Business Intelligence</p>
        <p>💡 <em>Ask natural language questions and get instant insights with beautiful visualizations</em></p>
    </div>
    """, 
    unsafe_allow_html=True
)
