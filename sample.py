import openai
import streamlit as st
import sqlite3
import os
import tempfile
import atexit
from sqlalchemy import create_engine, inspect, text, NullPool
from openai import OpenAI
from typing import List, Dict
import pandas as pd
import time
from urllib.parse import quote_plus
from dotenv import load_dotenv
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
import seaborn as sns

# Set your OpenAI API key (ensure you keep your API key safe!)
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

# Global variable to store temporary file paths
temp_files = [] 

def cleanup_temp_files():
    """Cleanup temporary files when the application exits."""
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error cleaning up {file_path}: {e}")

atexit.register(cleanup_temp_files)

def create_sqlalchemy_engine():
    """Create an SQLAlchemy engine using environment variables."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Fetch variables
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")
        dbname = os.getenv("dbname")
        
        # URL encode the password to handle special characters
        encoded_password = quote_plus(password)
        
        # Construct the connection URL
        database_url = f"postgresql+psycopg2://{user}:{encoded_password}@{host}:{port}/{dbname}"
        
        st.write("Debug: Using pooler connection")
        st.write(f"Debug: User: {user}")
        st.write(f"Debug: Host: {host}")
        
        return create_engine(
            database_url,
            connect_args={
                "sslmode": "require"
            }
        )
    except Exception as e:
        raise Exception(f"Engine creation failed: {str(e)}")

def get_db_schema(engine):
    """Get detailed database schema with sample data"""
    inspector = inspect(engine)
    schema = {}
    
    try:
        # Get list of all tables
        tables = inspector.get_table_names()
        if not tables:
            return {"error": "No tables found in database"}
            
        for table_name in tables:
            # Get column information
            inspector_columns = inspector.get_columns(table_name)
            column_info = []
            
            for col in inspector_columns:
                column_info.append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True)
                })
            
            # Get sample data and row count safely
            with engine.connect() as connection:
                # Get total row count
                count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                total_rows = connection.execute(count_query).scalar()
                
                # Get sample data
                sample_query = text(f"SELECT * FROM {table_name} LIMIT 5")
                result = connection.execute(sample_query)
                
                # Convert result to list of dictionaries without overriding the columns info
                sample_data = []
                result_columns = result.keys()
                for row in result:
                    sample_data.append(dict(zip(result_columns, row)))
                
                schema[table_name] = {
                    'columns': column_info,
                    'total_rows': total_rows,
                    'sample_data': sample_data
                }
        
        return schema
        
    except Exception as e:
        st.error(f"Error getting schema: {str(e)}")
        return {"error": str(e)}

def generate_response(prompt, db_context=None, query_result=None):
    """Generate a more natural response using OpenAI's API."""
    system_context = """You are a helpful and friendly database assistant. When responding:
    1. Always provide natural, conversational responses
    2. Format the response in a clear, engaging way using markdown
    3. When listing items, use bullet points or numbered lists appropriately
    4. Provide relevant context and insights about the data
    5. If showing numbers, format them in a readable way
    6. Never show raw SQL queries in the response
    7. Never use technical terms like "QUERY:" or "EXPLANATION:"
    8. Summarize the information in a human-friendly way
    9. If relevant, group or categorize the information logically
    10. Add helpful observations or patterns when appropriate
    11. Use a friendly, professional tone
    12. If the data is empty or null, explain that clearly
    13. For employee data, organize by departments or roles if applicable
    14. Add relevant emojis to make the response more engaging 👥 📊 ✨

    Example response style:
    "I found [number] employees in our database. Here they are, organized by department:

    **Sales Department** 👥
    • John Smith - Senior Sales Manager
    • Sarah Johnson - Sales Representative
    
    **Engineering Department** 💻
    • Mike Chen - Lead Developer
    • Ana Patel - Software Engineer

    [Additional insights or observations about the data]"
    """
    
    messages = [{"role": "system", "content": system_context}]
    
    if db_context:
        messages.append({"role": "system", "content": f"Database context:\n{db_context}"})
    
    if query_result is not None:
        messages.append({"role": "system", "content": f"Query result data: {query_result}"})
    
    messages.append({"role": "user", "content": prompt})
    
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content

def describe_database(schema):
    """Describe the database schema in natural language."""
    tables_info = "\n".join([f"Table '{table}' with columns: {', '.join(columns)}" for table, columns in schema.items()])
    prompt = f"The database has the following structure:\n\n{tables_info}\n\nProvide a natural language summary of this database."
    return generate_response(prompt)

def execute_query(engine, query):
    """Execute a query and return the result."""
    try:
        with engine.connect() as connection:
            # Ensure query is properly formatted as text
            query_text = text(query.strip())
            result = connection.execute(query_text)
            columns = result.keys()
            rows = result.fetchall()
            return columns, rows
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        raise e

def format_query_result(result, query):
    """Format query results into a pandas DataFrame with enhanced display"""
    try:
        columns, rows = result
        if not rows:
            return "No results found for this query."
            
        df = pd.DataFrame(rows, columns=columns)
        
        # Format datetime columns
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format numeric columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].round(2)
        
        return df
        
    except Exception as e:
        st.error(f"Error formatting results: {str(e)}")
        return f"Error formatting results: {str(e)}"

def clean_sql_query(query: str) -> str:
    """Clean SQL query by removing markdown formatting and unnecessary whitespace."""
    # Remove markdown SQL markers
    query = query.replace('```sql', '').replace('```', '')
    # Remove leading/trailing whitespace, semicolons, and empty lines
    query = '\n'.join(line.strip() for line in query.splitlines() if line.strip())
    # Remove trailing semicolon if present
    query = query.rstrip(';')
    return query

def main():
    st.title("DBBuddy: Your Smart Database Assistant")
    
    # Initialize session state for both database types
    if "chat_history_sqlite" not in st.session_state:
        st.session_state.chat_history_sqlite = []
    if "chat_history_supabase" not in st.session_state:
        st.session_state.chat_history_supabase = []
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False
    if "current_db_type" not in st.session_state:
        st.session_state.current_db_type = None

    # Database type selection
    new_db_type = st.radio("Choose your database type", ["SQLite", "Supabase"])
    
    # Handle database type switching
    if st.session_state.current_db_type != new_db_type:
        st.session_state.db_initialized = False
        st.session_state.current_db_type = new_db_type
    
    if new_db_type == "SQLite":
        display_sqlite_connection()
    elif new_db_type == "Supabase":
        display_supabase_connection()

    # Chat interface
    if st.session_state.db_initialized:
        display_chat_interface()

def display_supabase_connection():
    """Display Supabase connection interface"""
    st.info("""
    ### How to Connect to Supabase Database
    
    1. **Get Your Project Reference**:
       - Go to your Supabase project dashboard
       - Click on Project Settings (gear icon)
       - Your project reference is in the URL: https://[PROJECT-REF].supabase.co
    
    2. **Get Your Database Password**:
       - In Project Settings, go to Database
       - Look for "Connection Info" section
       - Find the Password field
       - Click "Generate New Password" if you don't have one
    
    3. **Connection Details**:
       - The connection will use the Transaction Pooler
       - Default host: aws-0-ap-south-1.pooler.supabase.com
       - Default port: 6543
       - Default database: postgres
    
    Note: Make sure you have the necessary permissions set up in your Supabase project.
    """)
    
    # Create .env file button
    col1, col2 = st.columns(2)
    
    with col1:
        project_ref = st.text_input(
            "Project Reference",
            placeholder="e.g., cavybxsgemqtptthfbya",
            help="Your Supabase project reference (found in project URL)"
        )
    
    with col2:
        password = st.text_input(
            "Database Password",
            type="password",
            help="Database password from Project Settings -> Database"
        )

    if st.button("Connect to Supabase"):
        try:
            # Create .env content
            env_content = f"""user=postgres.{project_ref}
password={password}
host=aws-0-ap-south-1.pooler.supabase.com
port=6543
dbname=postgres"""
            
            # Write to .env file
            with open('.env', 'w', newline='\n') as f:
                f.write(env_content.strip())
            
            # Try to connect
            engine = create_sqlalchemy_engine()
            schema = get_db_schema(engine)
            
            # Update session state
            st.session_state.db_engine = engine
            st.session_state.schema = schema
            st.session_state.db_initialized = True
            
            # Load previous chat history if exists
            load_chat_history()
            
            st.success("Successfully connected to Supabase database!")
            
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            st.session_state.db_initialized = False

def display_sqlite_connection():
    """Display SQLite connection interface"""
    st.info("""
    ### How to Connect to SQLite Database
    
    1. **Upload your SQLite database file**:
       - Prepare your .db or .sqlite file
       - Use the file uploader below
       - Maximum file size: 200MB
    
    2. **Or create a new database**:
       - Click "Create New Database" button
       - Enter a name for your database
    
    Note: Your database will be temporarily stored and removed when you close the application.
    """)
    
    # File uploader for existing database
    uploaded_file = st.file_uploader("Upload SQLite Database", type=['db', 'sqlite'])
    
    # Create new database option
    new_db_name = st.text_input("Or create a new database", placeholder="Enter database name (e.g., my_database.db)")
    create_new = st.button("Create New Database")
    
    if uploaded_file is not None:
        try:
            # Save uploaded file to temp location
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db.write(uploaded_file.getvalue())
            temp_db.close()
            
            # Connect to database
            engine = create_engine(f'sqlite:///{temp_db.name}')
            schema = get_db_schema(engine)
            
            # Update session state
            st.session_state.db_engine = engine
            st.session_state.schema = schema
            st.session_state.db_initialized = True
            st.session_state.current_temp_file = temp_db.name
            
            # Add to temp files list for cleanup
            temp_files.append(temp_db.name)
            
            # Load previous chat history if exists
            load_chat_history()
            
            st.success("Successfully connected to SQLite database!")
            
        except Exception as e:
            st.error(f"Error connecting to database: {str(e)}")
            st.session_state.db_initialized = False
            
    elif create_new and new_db_name:
        try:
            # Create new database file
            if not new_db_name.endswith('.db'):
                new_db_name += '.db'
            
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db.close()
            
            # Connect to new database
            engine = create_engine(f'sqlite:///{temp_db.name}')
            schema = get_db_schema(engine)
            
            # Update session state
            st.session_state.db_engine = engine
            st.session_state.schema = schema
            st.session_state.db_initialized = True
            st.session_state.current_temp_file = temp_db.name
            
            # Add to temp files list for cleanup
            temp_files.append(temp_db.name)
            
            # Load previous chat history if exists
            load_chat_history()
            
            st.success(f"Successfully created new SQLite database: {new_db_name}")
            
        except Exception as e:
            st.error(f"Error creating database: {str(e)}")
            st.session_state.db_initialized = False

def display_chat_interface():
    """Display the chat interface with RAG support"""
    # Initialize chat history if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your database..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response = generate_db_response(prompt)
            if response:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def save_chat_history():
    """Save chat history to a file"""
    history_file = (
        f"chat_history_{st.session_state.current_db_type.lower()}.json"
    )
    
    current_history = (st.session_state.chat_history_sqlite 
                      if st.session_state.current_db_type == "SQLite" 
                      else st.session_state.chat_history_supabase)
    
    # Convert DataFrame objects to dict for JSON serialization
    serializable_history = []
    for message in current_history:
        if isinstance(message["content"], pd.DataFrame):
            message["content"] = message["content"].to_dict()
        serializable_history.append(message)
    
    with open(history_file, 'w') as f:
        json.dump(serializable_history, f)

def load_chat_history():
    """Load chat history from file"""
    history_file = (
        f"chat_history_{st.session_state.current_db_type.lower()}.json"
    )
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
            
        # Convert dict back to DataFrame if needed
        for message in history:
            if isinstance(message["content"], dict):
                message["content"] = pd.DataFrame.from_dict(message["content"])
        
        if st.session_state.current_db_type == "SQLite":
            st.session_state.chat_history_sqlite = history
        else:
            st.session_state.chat_history_supabase = history

def create_database_context(schema):
    """Create a detailed context string from the database schema"""
    context = []
    
    for table_name, table_info in schema.items():
        # Table description
        table_desc = f"\nTable: {table_name}\n"
        table_desc += f"Total Records: {table_info['total_rows']}\n"
        
        # Column information
        table_desc += "Columns:\n"
        for col in table_info['columns']:
            table_desc += f"- {col['name']} ({col['type']}): "
            table_desc += f"{'NULL' if col['nullable'] else 'NOT NULL'}\n"
        
        # Sample data
        if table_info['sample_data']:
            sample_df = pd.DataFrame(table_info['sample_data'], 
                                   columns=[col['name'] for col in table_info['columns']])
            table_desc += f"\nSample Data:\n{sample_df.to_string()}\n"
            
            # Add basic statistics for numeric columns
            numeric_cols = sample_df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                stats = sample_df[numeric_cols].describe()
                table_desc += f"\nNumeric Column Statistics:\n{stats.to_string()}\n"
        
        context.append(table_desc)
    
    return "\n".join(context)

def setup_rag_system():
    """Setup the RAG system with database context"""
    try:
        # Create database context
        db_context = create_database_context(st.session_state.schema)
        
        # Split the context into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        context_chunks = text_splitter.split_text(db_context)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store
        vectorstore = FAISS.from_texts(context_chunks, embeddings)
        
        # Setup memory with output key
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # Create retrieval chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.7, model="gpt-4"),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": create_prompt_template()}
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error setting up RAG system: {str(e)}")
        return None

def create_prompt_template():
    """Create a custom prompt template for the RAG system"""
    template = """You are a helpful database assistant. Provide comprehensive, well-formatted responses.

    When analyzing data:
    1. Start with a clear summary
    2. Highlight key insights and patterns
    3. Mention any interesting trends or anomalies
    4. Suggest relevant visualizations when appropriate
    5. Format your response in clear sections
    6. Use markdown for better readability
    7. Include relevant statistics when useful

    If you need to query the database:
    QUERY: <the SQL query>
    EXPLANATION: <detailed analysis in markdown format>

    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    
    Answer: """
    
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

def analyze_database_structure(schema):
    """Analyze database structure and create metadata"""
    metadata = {
        'tables': {},
        'relationships': [],
        'data_types': set()
    }
    
    for table_name, info in schema.items():
        table_meta = {
            'columns': {},
            'numeric_columns': [],
            'text_columns': [],
            'date_columns': [],
            'categorical_columns': [],
            'row_count': info.get('total_rows', 0)
        }
        
        # Analyze columns
        for col in info.get('columns', []):
            col_type = str(col['type']).lower()
            col_name = col['name']
            
            table_meta['columns'][col_name] = {
                'type': col_type,
                'nullable': col.get('nullable', True)
            }
                
            # Categorize columns by type
            if any(t in col_type for t in ['int', 'float', 'decimal', 'numeric']):
                table_meta['numeric_columns'].append(col_name)
            elif any(t in col_type for t in ['varchar', 'text', 'char']):
                table_meta['text_columns'].append(col_name)
            elif any(t in col_type for t in ['date', 'time', 'timestamp']):
                table_meta['date_columns'].append(col_name)
            
            # Identify potential categorical columns
            if 'sample_data' in info:
                unique_values = len(set(row[col_name] for row in info['sample_data']))
                if unique_values < min(10, len(info['sample_data'])):
                    table_meta['categorical_columns'].append(col_name)
        
        metadata['tables'][table_name] = table_meta
    
    return metadata

def generate_dynamic_query(prompt, schema_metadata, table_name=None):
    """Generate SQL query based on natural language prompt and schema"""
    prompt_lower = prompt.lower()
    
    # If table not specified, try to determine from prompt and schema
    if table_name is None:
        for tbl in schema_metadata['tables']:
            if tbl.lower() in prompt_lower:
                table_name = tbl
                break
        
        if table_name is None:
            # Use first table as default if can't determine
            table_name = list(schema_metadata['tables'].keys())[0]
    
    table_meta = schema_metadata['tables'][table_name]
    
    # Determine query type
    is_aggregation = any(word in prompt_lower for word in ['average', 'highest', 'lowest', 'total', 'count'])
    is_comparison = any(word in prompt_lower for word in ['compare', 'difference', 'versus', 'vs'])
    is_trend = any(word in prompt_lower for word in ['trend', 'over time', 'change'])
    is_distribution = any(word in prompt_lower for word in ['distribution', 'spread', 'range'])
    
    # Build SELECT clause
    select_items = []
    group_by = []
    order_by = []
    where = []
    
    if is_aggregation:
        # Add relevant aggregations
        for col in table_meta['numeric_columns']:
            if 'total' in prompt_lower or 'sum' in prompt_lower:
                select_items.append(f"SUM({col}) as total_{col}")
            if 'average' in prompt_lower or 'mean' in prompt_lower:
                select_items.append(f"AVG({col}) as avg_{col}")
            if 'highest' in prompt_lower or 'maximum' in prompt_lower:
                select_items.append(f"MAX({col}) as max_{col}")
            if 'lowest' in prompt_lower or 'minimum' in prompt_lower:
                select_items.append(f"MIN({col}) as min_{col}")
        
        # Add grouping columns
        for col in table_meta['categorical_columns']:
            if col.lower() in prompt_lower:
                select_items.append(col)
                group_by.append(col)
    
    elif is_trend and table_meta['date_columns']:
        # Add date column and metrics
        date_col = table_meta['date_columns'][0]
        select_items.extend([date_col] + table_meta['numeric_columns'])
        order_by.append(date_col)
    
    elif is_distribution:
        # Select columns for distribution analysis
        select_items.extend(table_meta['numeric_columns'])
        if table_meta['categorical_columns']:
            select_items.extend(table_meta['categorical_columns'])
    
    else:
        # Default to selecting all columns
        select_items.append('*')
    
    # Build query
    query = f"SELECT {', '.join(select_items)} FROM {table_name}"
    
    if where:
        query += f" WHERE {' AND '.join(where)}"
    if group_by:
        query += f" GROUP BY {', '.join(group_by)}"
    if order_by:
        query += f" ORDER BY {', '.join(order_by)}"
    
    return query

def create_dynamic_visualization(df, query_type, metadata):
    """Create appropriate visualizations based on data and query type"""
    try:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        if query_type == 'trend' and not date_cols.empty:
            for date_col in date_cols:
                for num_col in numeric_cols:
                    st.markdown(f"#### {num_col} Trend")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df.plot(x=date_col, y=num_col, ax=ax)
                    st.pyplot(fig)
                    plt.close()
        
        elif query_type == 'distribution' and not numeric_cols.empty:
            for col in numeric_cols:
                st.markdown(f"#### {col} Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x=col, kde=True)
                st.pyplot(fig)
                plt.close()
        
        elif query_type == 'comparison' and not categorical_cols.empty:
            for cat_col in categorical_cols:
                for num_col in numeric_cols:
                    st.markdown(f"#### {num_col} by {cat_col}")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=df, x=cat_col, y=num_col)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def analyze_query_intent(prompt: str) -> dict:
    """Analyze the query intent without assumptions about schema"""
    return {
        'comparison': any(word in prompt.lower() for word in ['highest', 'lowest', 'most', 'least', 'top', 'bottom']),
        'aggregation': any(word in prompt.lower() for word in ['average', 'total', 'count', 'sum']),
        'fields_of_interest': extract_fields_of_interest(prompt),
        'raw_prompt': prompt.lower()
    }

def extract_fields_of_interest(prompt: str) -> list:
    """Extract potential field names from prompt"""
    common_fields = ['name', 'salary', 'position', 'department', 'date', 'id', 'email', 'phone']
    return [field for field in common_fields if field in prompt.lower()]

def build_dynamic_query(intent: dict, schema: dict) -> tuple:
    """Build a SQL query based on intent and available schema"""
    # Find relevant table
    relevant_table = find_relevant_table(schema, intent['fields_of_interest'])
    if not relevant_table:
        return None, "Could not determine relevant table"

    table_info = schema[relevant_table]
    columns = [col['name'] for col in table_info['columns']]
    
    # Build SELECT clause
    select_clause = []
    order_clause = []
    
    # If comparison is needed, find comparable columns
    if intent['comparison']:
        numeric_cols = [col['name'] for col in table_info['columns'] 
                       if any(t in str(col['type']).lower() for t in ['int', 'float', 'decimal', 'numeric'])]
        if numeric_cols:
            select_clause.extend(columns)  # Select all columns for context
            order_clause.extend([f"{col} DESC" for col in numeric_cols])
    else:
        select_clause.extend(columns)
    
    # Build query
    query = f"SELECT {', '.join(select_clause) or '*'} FROM {relevant_table}"
    if order_clause:
        query += f" ORDER BY {', '.join(order_clause)}"
    query += " LIMIT 100"  # Safe limit
    
    return query, relevant_table

def find_relevant_table(schema: dict, fields: list) -> str:
    """Find most relevant table based on field matches"""
    best_match = None
    max_matches = 0
    
    for table, info in schema.items():
        column_names = [col['name'].lower() for col in info['columns']]
        matches = sum(1 for field in fields if field in column_names)
        if matches > max_matches:
            max_matches = matches
            best_match = table
    
    return best_match or next(iter(schema))  # Return first table if no match

def analyze_results(df: pd.DataFrame, intent: dict) -> dict:
    """Analyze query results based on data types"""
    analysis = {
        'numeric_analysis': {},
        'categorical_analysis': {},
        'temporal_analysis': {}
    }
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            analysis['numeric_analysis'][column] = {
                'mean': float(df[column].mean()),  # Convert to Python float
                'max': float(df[column].max()),    # Convert to Python float
                'min': float(df[column].min())     # Convert to Python float
            }
        elif df[column].dtype == 'object':
            unique_values = df[column].nunique()
            if unique_values < len(df) * 0.5:  # Potential categorical
                # Convert Series to dictionary with Python int values
                value_counts = df[column].value_counts()
                analysis['categorical_analysis'][column] = {
                    str(k): int(v) for k, v in value_counts.items()
                }
        elif 'datetime' in str(df[column].dtype):
            analysis['temporal_analysis'][column] = {
                'earliest': str(df[column].min()),  # Convert datetime to string
                'latest': str(df[column].max())    # Convert datetime to string
            }
    
    return analysis

def create_dynamic_visualizations(df: pd.DataFrame, analysis: dict, intent: dict):
    """Create visualizations based on data types and intent"""
    for col, stats in analysis['numeric_analysis'].items():
        if intent['comparison']:
            st.markdown(f"#### Distribution of {col}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=col, kde=True)
            plt.axvline(stats['max'], color='red', linestyle='--', label='Maximum')
            plt.axvline(stats['mean'], color='green', linestyle='--', label='Average')
            plt.legend()
            st.pyplot(fig)
            plt.close()
    
    for col, counts in analysis['categorical_analysis'].items():
        st.markdown(f"#### {col} Distribution")
        st.bar_chart(counts)
    
    # Create correlation matrix for numeric columns if multiple exist
    numeric_cols = list(analysis['numeric_analysis'].keys())
    if len(numeric_cols) > 1:
        st.markdown("#### Correlation Matrix")
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        st.pyplot(fig)
        plt.close()

def generate_natural_response(df: pd.DataFrame, analysis: dict, intent: dict) -> str:
    """Generate natural language response based on analysis"""
    response = []
    
    if intent['comparison']:
        for col, stats in analysis['numeric_analysis'].items():
            max_row = df[df[col] == stats['max']].iloc[0]
            response.append(f"Highest {col}: {stats['max']:,.2f}")
            for other_col in df.columns:
                if other_col != col:
                    response.append(f"- {other_col}: {max_row[other_col]}")
    
    return "\n".join(response)

def refine_chatgpt_response(prompt: str, df: pd.DataFrame, analysis: dict, query: str) -> str:
    """Generate a refined ChatGPT-style response incorporating text, table, analysis summary, and full database context."""
    try:
        # Convert DataFrame to a more readable format for the LLM
        if not df.empty:
            data_description = df.to_dict(orient='records')
        else:
            data_description = "No data available"
        
        # Create a context that encourages more natural responses
        context = f"""Based on the database query results, provide a natural, conversational response.
        
Data Results:
{data_description}

Additional Context:
- Total records found: {len(df) if not df.empty else 0}
- Columns available: {', '.join(df.columns) if not df.empty else 'None'}
{create_database_context(st.session_state.schema)}

Please provide a response that:
1. Starts with a friendly introduction
2. Lists the information in a clear, organized way
3. Groups related information when possible
4. Adds relevant insights or patterns
5. Uses appropriate formatting and emojis
6. Maintains a conversational tone
7. Never mentions SQL or queries
8. Provides a natural conclusion"""

        refined_response = generate_response(prompt, db_context=context)
        return refined_response
        
    except Exception as e:
        return f"I apologize, but I encountered an error while processing the data: {str(e)}"

def generate_db_response(prompt: str):
    """Main function to generate the database response with refinements and multimodal outputs.
    
    This function:
      - Analyzes the query intent.
      - Dynamically builds and executes the SQL query.
      - Formats results into a pandas DataFrame.
      - Displays interactive components (tables and graphs) via Streamlit.
      - Uses the refined response generator to create a final answer which includes a clear
        text summary comparable to ChatGPT responses.
    """
    try:
        # Analyze query intent
        intent = analyze_query_intent(prompt)
        
        # Build and execute the dynamic query
        query, table = build_dynamic_query(intent, st.session_state.schema)
        if not query:
            return "Could not determine how to query the database."
        
        result = execute_query(st.session_state.db_engine, query)
        df = format_query_result(result, query)
        
        # If an error message string is returned instead of a DataFrame
        if isinstance(df, str):
            return df
        
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Perform basic analysis on the results
            analysis = analyze_results(df, intent)
            
            # Display interactive components
            st.markdown("### Query Results")
            st.dataframe(df)
            create_dynamic_visualizations(df, analysis, intent)
            
            # Generate a refined response that includes text, tables, and analysis
            refined_text = refine_chatgpt_response(prompt, df, analysis, query)
            
            # Combine the refined text with a clear header into the final response
            final_response = f"**Final Analysis:**\n\n{refined_text}"
            return final_response
        else:
            return "No data found matching your query."
            
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return f"Error: {str(e)}"

def display_advanced_statistics(df):
    """Display advanced statistical analysis"""
    st.markdown("### 📊 Statistical Analysis")
    
    # Basic statistics
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if not numeric_cols.empty:
        st.markdown("#### 📈 Descriptive Statistics")
        st.dataframe(df[numeric_cols].describe())
        
        # Additional statistics
        for col in numeric_cols:
            st.markdown(f"#### 📊 {col.title()} Analysis")
            stats = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Mode', 'Std Dev', 'Skewness', 'Kurtosis'],
                'Value': [
                    df[col].mean(),
                    df[col].median(),
                    df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    df[col].std(),
                    df[col].skew(),
                    df[col].kurtosis()
                ]
            })
            st.dataframe(stats)

def describe_database_contents(schema):
    """Generate a comprehensive database description"""
    if "error" in schema:
        return f"Error accessing database: {schema['error']}"
        
    description = "### 📚 Database Overview\n\n"
    
    if not schema:
        return description + "No tables found in the database."
    
    # First, give a summary of all tables
    description += "This database contains the following tables:\n\n"
    for table_name, info in schema.items():
        description += f"- **{table_name}** ({info['total_rows']} records)\n"
    description += "\n"
    
    # Then, detail each table
    for table_name, table_info in schema.items():
        description += f"### 📑 Table: {table_name}\n"
        description += f"Total Records: {table_info['total_rows']}\n\n"
        
        # Column information
        description += "#### Columns:\n"
        for col in table_info['columns']:
            description += f"- **{col['name']}** ({col['type']})"
            description += f" {'(Optional)' if col['nullable'] else '(Required)'}\n"
        
        description += "\n#### Sample Data:\n"
        if table_info['sample_data']:
            try:
                # Convert sample data to DataFrame for better display
                df = pd.DataFrame(table_info['sample_data'])
                if not df.empty:
                    st.markdown(description)
                    styled_df = style_dataframe(df)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Show basic statistics for numeric columns
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    if not numeric_cols.empty:
                        st.markdown("#### 📊 Basic Statistics")
                        st.dataframe(df[numeric_cols].describe())
                    
                    # Show value counts for categorical columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if not categorical_cols.empty:
                        st.markdown("#### 📊 Category Distributions")
                        for col in categorical_cols:
                            st.markdown(f"**{col}** distribution:")
                            st.dataframe(df[col].value_counts())
                    
                description = ""  # Clear description as we've displayed it
            except Exception as e:
                description += f"Error displaying sample data: {str(e)}\n\n"
        else:
            description += "No sample data available\n\n"
    
    return description

def style_dataframe(df):
    """Apply consistent styling to DataFrame"""
    return df.style.set_properties(**{
        'background-color': '#f0f2f6',
        'color': 'black',
        'border-color': '#e1e4e8',
        'padding': '8px'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4a90e2'), 
                                   ('color', 'white'),
                                   ('font-weight', 'bold')]},
        {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', '#f9f9f9')]},
    ])

def display_data_insights(df):
    """Display comprehensive data insights"""
    st.markdown("### 📊 Data Insights")
    
    # Basic statistics
    if not df.empty:
        # Numeric analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if not numeric_cols.empty:
            st.markdown("#### 📈 Numeric Data Analysis")
            st.dataframe(df[numeric_cols].describe())
            
            # Correlation matrix for numeric columns
            if len(numeric_cols) > 1:
                st.markdown("#### 🔄 Correlation Matrix")
                corr_matrix = df[numeric_cols].corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            st.markdown("#### 📊 Categorical Data Analysis")
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                st.markdown(f"**{col}** distribution:")
                st.dataframe(pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(df) * 100).round(2)
                }))
        
        # Date analysis
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if not date_cols.empty:
            st.markdown("#### 📅 Temporal Analysis")
            for col in date_cols:
                st.markdown(f"**{col}** timeline:")
                st.line_chart(df[col].value_counts().sort_index())

def display_visualizations(df):
    """Display interactive visualizations based on data types"""
    st.markdown("### 📊 Data Visualizations")
    
    if not df.empty:
        # Numeric visualizations
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if not numeric_cols.empty:
            st.markdown("#### 📈 Numeric Data Visualization")
            
            # Distribution plots
            for col in numeric_cols:
                st.markdown(f"**{col}** distribution:")
                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.bar_chart(df[col].value_counts())
                with fig_col2:
                    st.line_chart(df[col].value_counts())
        
        # Categorical visualizations
        categorical_cols = df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            st.markdown("#### 📊 Categorical Data Visualization")
            for col in categorical_cols:
                st.markdown(f"**{col}** distribution:")
                st.bar_chart(df[col].value_counts())
        
        # Time series visualizations
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if not date_cols.empty:
            st.markdown("#### 📅 Time Series Visualization")
            for date_col in date_cols:
                st.markdown(f"**{date_col}** timeline:")
                st.line_chart(df.set_index(date_col))

def display_employee_statistics(df):
    """Display employee-specific statistics and insights"""
    st.markdown("#### 📊 Employee Statistics")
    
    # Total employee count
    total_employees = len(df)
    st.metric("Total Employees", total_employees)
    
    # Department/Position distribution if available
    if 'position' in df.columns or 'department' in df.columns:
        position_col = 'position' if 'position' in df.columns else 'department'
        st.markdown(f"#### 👥 Distribution by {position_col.title()}")
        position_dist = df[position_col].value_counts()
        st.bar_chart(position_dist)
    
    # Salary statistics if available
    if 'salary' in df.columns:
        st.markdown("#### 💰 Salary Overview")
        salary_stats = df['salary'].describe()
        st.dataframe(salary_stats)
        
        # Salary distribution
        st.markdown("#### 📈 Salary Distribution")
        fig, ax = plt.subplots()
        df['salary'].hist(bins=20, ax=ax)
        st.pyplot(fig)
    
    # Experience/tenure if available
    if 'start_date' in df.columns:
        st.markdown("#### 📅 Employee Tenure")
        df['tenure'] = pd.to_datetime('now') - pd.to_datetime(df['start_date'])
        tenure_years = df['tenure'].dt.total_seconds() / (365.25 * 24 * 60 * 60)
        st.metric("Average Tenure (years)", f"{tenure_years.mean():.1f}")

if __name__ == "__main__":
    main()