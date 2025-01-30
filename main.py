import streamlit as st
import sqlite3
import os
import tempfile
import atexit
from sqlalchemy import create_engine, inspect, text
import openai
from typing import List, Dict
import pandas as pd

# Set your OpenAI API key
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

def create_sqlalchemy_engine(db_path):
    """Create an SQLAlchemy engine for SQLite."""
    return create_engine(f"sqlite:///{db_path}")

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
    5. When a query is needed, format your response as follows:
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

def format_query_result(rows: List, query: str) -> pd.DataFrame:
    """Format query results into a pandas DataFrame for better display"""
    try:
        df = pd.DataFrame(rows)
        if not df.empty:
            return df
        return "No results found for this query."
    except Exception as e:
        return f"Error formatting results: {str(e)}"

def clean_sql_query(query: str) -> str:
    """Clean SQL query by removing markdown formatting and unnecessary whitespace."""
    # Remove markdown SQL markers
    query = query.replace('```sql', '').replace('```', '')
    # Remove leading/trailing whitespace and empty lines
    query = '\n'.join(line.strip() for line in query.splitlines() if line.strip())
    return query

def main():
    st.title("DBBuddy: Your Smart Database Assistant")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False

    # File uploader
    uploaded_file = st.file_uploader("Upload your SQLite database file", type=["db", "sqlite", "sqlite3"])
    
    if uploaded_file and not st.session_state.db_initialized:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_file:
            temp_file_path = temp_file.name
            temp_files.append(temp_file_path)
            temp_file.write(uploaded_file.getbuffer())

        try:
            engine = create_sqlalchemy_engine(temp_file_path)
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
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for chat in st.session_state.chat_history:
            role = "You" if chat["role"] == "user" else "DBBuddy"
            message = chat["content"]
            
            # If the message is a DataFrame, display it as a table
            if isinstance(message, pd.DataFrame):
                st.markdown(f"**{role}:**")
                st.dataframe(message)
            else:
                st.markdown(f"**{role}:** {message}")

    # User input
    if st.session_state.db_initialized:
        user_input = st.chat_input("Ask me anything about your database...")
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            try:
                engine = st.session_state.db_engine
                schema = st.session_state.schema

                # Generate response with context
                response = generate_response(
                    f"Based on this database schema: {schema}, the user asks: {user_input}. "
                    "If this requires a SQL query, format your response with QUERY: followed by the SQL query, "
                    "then EXPLANATION: followed by your explanation. Do not use markdown formatting in the SQL query.",
                    schema
                )
                
                # Check if response contains SQL query
                if "QUERY:" in response:
                    # Extract query and explanation
                    parts = response.split("QUERY:", 1)[1].split("EXPLANATION:")
                    query = clean_sql_query(parts[0].strip())
                    explanation = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Execute query
                    result = execute_query(engine, query)
                    
                    # Format results
                    df_result = format_query_result(result, query)
                    
                    # Generate natural language explanation of results
                    final_explanation = generate_response(
                        f"Explain these query results in natural language: {result}",
                        schema,
                        result
                    )
                    
                    # Add results and explanation to chat
                    if isinstance(df_result, pd.DataFrame):
                        st.session_state.chat_history.append({"role": "assistant", "content": df_result})
                    st.session_state.chat_history.append({"role": "assistant", "content": final_explanation})
                else:
                    # Add regular response to chat
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Rerun the app to update the chat
                st.rerun()
                
            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.rerun()
    else:
        st.info("Please upload a database file to start the conversation.")

if __name__ == "__main__":
    main()
