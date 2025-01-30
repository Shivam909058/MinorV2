import streamlit as st
import os
import tempfile
import atexit
from sqlalchemy import create_engine, inspect, text
import openai
import pandas as pd
from typing import List

# Set your OpenAI API key
OPENAI_API_KEY = ""  # Use your OpenAI key
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

def create_sqlalchemy_engine(database_url):
    """Create an SQLAlchemy engine for any database using its connection URL."""
    try:
        engine = create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
        return engine
    except Exception as e:
        raise Exception(f"Failed to connect to database: {str(e)}")

def get_db_schema(engine):
    """Retrieve schema details from the database."""
    inspector = inspect(engine)
    schema = {}
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        schema[table_name] = [col['name'] for col in columns]
    return schema

def generate_response(prompt, schema=None, query_result=None):
    """Generate a more natural response using OpenAI's API."""
    system_context = """You are a helpful database assistant. When responding:
    1. If providing query results, explain them in natural language
    2. Format numbers and data clearly
    3. Provide insights about the data when possible
    4. Be conversational but professional
    5. For questions about data counts or table information, ALWAYS include a SQL query
    6. When a query is needed, format your response as follows:
       QUERY: <write the SQL query here>
       EXPLANATION: <write your explanation here>"""
    
    messages = [{"role": "system", "content": system_context}]
    
    if schema:
        schema_context = f"Database schema: {schema}"
        messages.append({"role": "system", "content": schema_context})
    
    if query_result is not None:
        result_context = f"Query result: {query_result}"
        messages.append({"role": "system", "content": result_context})
    
    messages.append({"role": "user", "content": prompt})
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

def describe_database(schema):
    """Describe the database schema in natural language."""
    tables_info = "\n".join([f"Table '{table}' with columns: {', '.join(columns)}" for table, columns in schema.items()])
    prompt = f"The database has the following structure:\n\n{tables_info}\n\nProvide a natural language summary of this database."
    return generate_response(prompt)

def execute_query(engine, query):
    """Execute a query and return the result."""
    with engine.connect() as connection:
        result = connection.execute(text(query))
        rows = result.fetchall()
    return rows

def format_query_result(rows: List) -> pd.DataFrame:
    """Format query results into a pandas DataFrame for better display."""
    try:
        df = pd.DataFrame(rows)
        if not df.empty and len(df.columns) > 0:
            # If columns are numeric indices, try to get column names from first row
            if all(isinstance(col, int) for col in df.columns):
                df.columns = rows[0].keys()
        return df
    except Exception as e:
        return f"Error formatting results: {str(e)}"

def clean_sql_query(query: str) -> str:
    """Clean SQL query by removing markdown formatting and unnecessary whitespace."""
    query = query.replace('```sql', '').replace('```', '')
    query = '\n'.join(line.strip() for line in query.splitlines() if line.strip())
    return query

def main():
    st.title("DBBuddy: Your Smart Database Assistant")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False

    db_type = st.radio("Choose your database type", ["SQLite", "Supabase"])
    
    if db_type == "Supabase":
        st.markdown("""
        ### Supabase Connection String Format:
        ```
        postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres
        ```
        """)
        db_url = st.text_input("Enter your Supabase PostgreSQL connection URL", type="password")
        if db_url:
            try:
                with st.spinner('Connecting to database...'):
                    engine = create_sqlalchemy_engine(db_url)
                    schema = get_db_schema(engine)
                    st.session_state.db_engine = engine
                    st.session_state.schema = schema
                    st.session_state.db_initialized = True
                    
                    # Generate initial database description
                    welcome_msg = describe_database(schema)
                    st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
                    st.success("Successfully connected to database!")
                    
            except Exception as e:
                st.error(f"Error connecting to database: {str(e)}")

    if db_type == "SQLite":
        uploaded_file = st.file_uploader("Upload your SQLite database file", type=["db", "sqlite", "sqlite3"])
        if uploaded_file and not st.session_state.db_initialized:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_file:
                temp_file_path = temp_file.name
                temp_files.append(temp_file_path)
                temp_file.write(uploaded_file.getbuffer())

            try:
                engine = create_sqlalchemy_engine(f"sqlite:///{temp_file_path}")
                schema = get_db_schema(engine)
                st.session_state.db_engine = engine
                st.session_state.schema = schema
                st.session_state.db_initialized = True
                
                # Generate initial database description
                welcome_msg = describe_database(schema)
                st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
                
            except Exception as e:
                st.error(f"Error initializing database: {e}")

    # Chat interface
    if st.session_state.db_initialized:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write("You: " + message["content"])
            else:
                # Check if content is DataFrame
                if isinstance(message["content"], pd.DataFrame):
                    st.write("Assistant: Here are the results:")
                    st.dataframe(message["content"])
                else:
                    st.write("Assistant: " + str(message["content"]))

        user_input = st.chat_input("Ask me anything about your database...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                engine = st.session_state.db_engine
                schema = st.session_state.schema

                # Generate response with context
                response = generate_response(f"Based on this database schema: {schema}, {user_input}")
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Execute query if present
                if "QUERY:" in response:
                    parts = response.split("QUERY:", 1)[1].split("EXPLANATION:")
                    query = clean_sql_query(parts[0].strip())
                    explanation = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Execute and display query results
                    result = execute_query(engine, query)
                    formatted_result = format_query_result(result)
                    
                    # Add results to chat history
                    if isinstance(formatted_result, pd.DataFrame):
                        st.session_state.chat_history.append({"role": "assistant", "content": formatted_result})
                        if explanation:
                            st.session_state.chat_history.append({"role": "assistant", "content": explanation})
                
                st.rerun()
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.rerun()
    else:
        st.info("Please upload your database or provide a connection URL.")

if __name__ == "__main__":
    main()
