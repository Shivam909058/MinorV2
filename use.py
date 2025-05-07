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
import random
import datetime
from supabase import create_client
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import bcrypt
import re
from functools import lru_cache

# Set your OpenAI API key (ensure you keep your API key safe!)
# Replace with environment variable or secure method
openai.api_key = ""

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
    """Generate a more natural response using OpenAI's API with reduced token usage."""
    system_context = """You are a helpful database assistant. Be concise and direct. Format with markdown when helpful.
    Avoid technical SQL terms. Summarize information clearly with bullet points when appropriate. 
    Be friendly but efficient with words. If data is empty, state so clearly. Use emojis sparingly."""
    
    messages = [{"role": "system", "content": system_context}]
    
    # Optimize context by limiting size
    if db_context:
        # Truncate context if too large (over 2000 chars)
        if len(db_context) > 2000:
            db_context = db_context[:1900] + "...[truncated for brevity]"
        messages.append({"role": "system", "content": f"Database context:\n{db_context}"})
    
    # Optimize query result data
    if query_result is not None:
        # Limit the amount of data sent to the API
        if isinstance(query_result, list) and len(query_result) > 10:
            query_result = query_result[:10] + ["...and more (truncated)"]
        messages.append({"role": "system", "content": f"Query result data (sample): {query_result}"})
    
    messages.append({"role": "user", "content": prompt})
    
    client = OpenAI()
    
    # Use a more efficient model and limit token output
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use 3.5-turbo instead of GPT-4 to reduce costs
        messages=messages,
        temperature=0.5,
        max_tokens=250  # Limit response length
    )
    return response.choices[0].message.content

def describe_database(schema):
    """Describe the database schema in natural language."""
    tables_info = "\n".join([f"Table '{table}' with columns: {', '.join(columns)}" for table, columns in schema.items()])
    prompt = f"The database has the following structure:\n\n{tables_info}\n\nProvide a natural language summary of this database."
    return generate_response(prompt)

def execute_query(connection, query):
    """Universal query execution function that works with any connection type"""
    try:
        # Handle connection not being available
        if connection is None:
            return ["error"], [("No database connection available.",)]
            
        # Determine connection type
        if "connection_type" in st.session_state:
            connection_type = st.session_state.connection_type
        else:
            # Default to SQLite for backward compatibility
            connection_type = "sqlite"
        
        # Handle REST API connection (Supabase client)
        if connection_type == "rest" and hasattr(connection, 'rpc'):
            try:
                # First try running through RPC if available
                response = connection.rpc('run_query', {'query_text': query}).execute()
                
                if hasattr(response, 'data') and response.data:
                    # Get results - for JSON responses from RPC
                    if len(response.data) > 0:
                        # Extract column names from the first result
                        columns = list(response.data[0].keys())
                        # Convert to rows
                        rows = [tuple(row.values()) for row in response.data]
                        return columns, rows
                    else:
                        return [], []
                        
            except Exception as rpc_error:
                # If RPC fails, try to parse the query and use direct table access
                table_name, conditions = parse_sql_for_rest_api(query)
                
                if not table_name:
                    # Final fallback - try the simplified approach
                    try:
                        # Get any table from schema and return first few rows
                        if 'schema' in st.session_state and st.session_state.schema:
                            # Get the first table that's not a system table
                            for t_name in st.session_state.schema:
                                if not t_name.startswith(('pg_', 'auth_')):
                                    fallback_response = connection.table(t_name).select('*').limit(10).execute()
                                    if hasattr(fallback_response, 'data') and fallback_response.data:
                                        columns = list(fallback_response.data[0].keys())
                                        rows = [tuple(row.values()) for row in fallback_response.data]
                                        return columns, rows
                                    break
                    except:
                        pass
                        
                    return ["error"], [(f"Could not execute query via REST API. Error: {str(rpc_error)}",)]
                
                try:
                    # Build Supabase query with proper table method
                    supabase_query = connection.table(table_name)
                    
                    # Apply conditions if any
                    if conditions.get('select') and conditions['select'] != ['*']:
                        supabase_query = supabase_query.select(','.join(conditions['select']))
                    else:
                        # Default to selecting all columns
                        supabase_query = supabase_query.select('*')
                        
                    if conditions.get('where'):
                        for condition in conditions['where']:
                            if len(condition) == 3:
                                column, operator, value = condition
                                if operator == '=':
                                    supabase_query = supabase_query.eq(column, value)
                                elif operator == '!=':
                                    supabase_query = supabase_query.neq(column, value)
                                elif operator == '>':
                                    supabase_query = supabase_query.gt(column, value)
                                elif operator == '<':
                                    supabase_query = supabase_query.lt(column, value)
                    
                    # Apply limit
                    if conditions.get('limit'):
                        supabase_query = supabase_query.limit(conditions['limit'])
                    else:
                        supabase_query = supabase_query.limit(100)  # Default limit
                    
                    # Apply order
                    if conditions.get('order'):
                        for col, direction in conditions['order']:
                            ascending = direction.lower() != 'desc'
                            supabase_query = supabase_query.order(col, ascending=ascending)
                    
                    # Execute the query
                    response = supabase_query.execute()
                    
                    if hasattr(response, 'data') and response.data:
                        if len(response.data) > 0:
                            # Get column names from the first row
                            columns = list(response.data[0].keys())
                            # Convert data to row tuples
                            rows = [tuple(row.values()) for row in response.data]
                            return columns, rows
                        else:
                            # Return empty result with column names if possible
                            test_response = connection.table(table_name).select('*').limit(1).execute()
                            if hasattr(test_response, 'data') and test_response.data and len(test_response.data) > 0:
                                columns = list(test_response.data[0].keys())
                                return columns, []
                            else:
                                return [], []
                    else:
                        return [], []
                        
                except Exception as direct_error:
                    # Create a meaningful error message
                    return ["error"], [(f"Error querying table '{table_name}': {str(direct_error)}",)]
        
        # Handle PostgreSQL direct connection or SQLite
        else:
            with connection.connect() as conn:
                # Clean up the query
                query_text = text(query.strip())
                
                try:
                    # Execute the query safely
                    result = conn.execute(query_text)
                    columns = result.keys()
                    rows = result.fetchall()
                    return columns, rows
                except Exception as e:
                    # Try to handle "no such column" errors
                    error_msg = str(e)
                    if "no such column" in error_msg.lower():
                        # Extract table name and retry with * query
                        import re
                        table_match = re.search(r'FROM\s+(["\w]+)', query, re.IGNORECASE)
                        if table_match:
                            table_name = table_match.group(1).replace('"', '')
                            try:
                                fallback_query = text(f'SELECT * FROM "{table_name}" LIMIT 100')
                                result = conn.execute(fallback_query)
                                columns = result.keys()
                                rows = result.fetchall()
                                return columns, rows
                            except:
                                pass
                    # Return the error message in a format that can be displayed
                    return ["error"], [(f"SQL Error: {error_msg}",)]
                    
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        # Return error message in a format that works with the dataframe display
        return ["error"], [(f"Error: {str(e)}",)]

def parse_sql_for_rest_api(query):
    """Parse a SQL query into components for REST API usage with improved reliability"""
    import re
    
    # Default conditions
    conditions = {
        'select': [],
        'where': [],
        'limit': 100,
        'order': []
    }
    
    try:
        # Extract table name - handle quoted identifiers
        table_match = re.search(r'FROM\s+(["\w\.]+)', query, re.IGNORECASE)
        if not table_match:
            return None, conditions
        
        table_name = table_match.group(1).replace('"', '')
        
        # Handle schema qualified names (extract just the table part)
        if '.' in table_name:
            parts = table_name.split('.')
            if len(parts) == 2 and parts[0] in ('public', 'information_schema'):
                table_name = parts[1]
        
        # Extract SELECT columns
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE)
        if select_match:
            select_part = select_match.group(1).strip()
            if select_part == '*':
                conditions['select'] = ['*']
            else:
                # Split and clean quoted identifiers
                columns = []
                for col in select_part.split(','):
                    col = col.strip().replace('"', '')
                    # Remove table alias prefixes (t.column_name → column_name)
                    if '.' in col:
                        col = col.split('.')[-1]
                    columns.append(col)
                conditions['select'] = columns
        
        # Extract WHERE conditions with improved regex
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_part = where_match.group(1).strip()
            
            # Handle equality conditions
            equality_conditions = re.findall(r'(\w+)\s*=\s*[\'"]?([^\'";\s]+)[\'"]?', where_part)
            for column, value in equality_conditions:
                conditions['where'].append((column, '=', value))
            
            # Handle inequality conditions
            inequality_conditions = re.findall(r'(\w+)\s*([<>!]=?)\s*[\'"]?([^\'";\s]+)[\'"]?', where_part)
            for column, operator, value in inequality_conditions:
                if operator not in ('=', '=='):  # Skip equals already covered
                    conditions['where'].append((column, operator, value))
        
        # Extract LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            conditions['limit'] = int(limit_match.group(1))
        
        # Extract ORDER BY
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|$)', query, re.IGNORECASE)
        if order_match:
            order_part = order_match.group(1).strip()
            order_columns = order_part.split(',')
            for order_col in order_columns:
                parts = order_col.strip().split()
                col = parts[0].replace('"', '')
                # Handle table alias prefixes (t.column_name → column_name)
                if '.' in col:
                    col = col.split('.')[-1]
                direction = parts[1] if len(parts) > 1 else 'ASC'
                conditions['order'].append((col, direction))
        
        return table_name, conditions
    except Exception:
        # Return None on any parsing error
        return None, conditions

def format_query_result(result, query):
    """Format query results into a pandas DataFrame with enhanced display"""
    try:
        columns, rows = result
        
        # Check if this is an error message
        if len(columns) == 1 and columns[0] == "error":
            error_message = rows[0][0] if rows and len(rows) > 0 else "Unknown error"
            st.error(f"Query error: {error_message}")
            return pd.DataFrame({"Error": [error_message]})
            
        if not rows:
            return pd.DataFrame(columns=columns) if columns else "No results found for this query."
            
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
        return pd.DataFrame({"Error": [f"Error formatting results: {str(e)}"]})

def clean_sql_query(query: str) -> str:
    """Clean SQL query by removing markdown formatting and unnecessary whitespace."""
    # Remove markdown SQL markers
    query = query.replace('```sql', '').replace('```', '')
    # Remove leading/trailing whitespace, semicolons, and empty lines
    query = '\n'.join(line.strip() for line in query.splitlines() if line.strip())
    # Remove trailing semicolon if present
    query = query.rstrip(';')
    return query

def initialize_authentication():
    """Set up authentication for the application using Streamlit session state"""
    # Check if authentication cookie exists
    if 'authentication_status' not in st.session_state:
        # Try to load from cookies if not in session
        # Streamlit doesn't have direct cookie API, so we'll use an alternative approach
        # with session persistence
        
        # Set session state variables
        st.session_state.authentication_status = None
        st.session_state.username = None
        st.session_state.name = None
        st.session_state.role = None
        st.session_state.full_access = False
        
        # Try loading from saved state file if exists
        try:
            if os.path.exists('.streamlit/session_state.json'):
                with open('.streamlit/session_state.json', 'r') as f:
                    saved_state = json.load(f)
                    if saved_state.get('auth_valid_until') and float(saved_state['auth_valid_until']) > time.time():
                        # Session still valid
                        st.session_state.authentication_status = True
                        st.session_state.username = saved_state.get('username')
                        st.session_state.name = saved_state.get('name')
                        st.session_state.role = saved_state.get('role')
                        st.session_state.full_access = saved_state.get('full_access', False)
        except Exception as e:
            # If any error occurs, continue with regular authentication
            pass
    
    # Load configuration file with user credentials
    # If file doesn't exist, create default config
    try:
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
            
        # Check if first admin account exists, if not add it
        if 'admin_123' not in config['credentials']['usernames']:
            config['credentials']['usernames']['admin_123'] = {
                'email': 'shivamatvit@gmail.com',
                'name': 'Primary Admin',
                'password': bcrypt.hashpw('aaddmin@123'.encode(), bcrypt.gensalt()).decode(),
                'role': 'admin',
                'full_access': True
            }
            
        # Check if second admin account exists, if not add it
        if 'system_administrator_2023' not in config['credentials']['usernames']:
            config['credentials']['usernames']['system_administrator_2023'] = {
                'email': 'admin@dbbuddy.com',
                'name': 'System Administrator',
                'password': bcrypt.hashpw('ComplexP@$$w0rd2023!'.encode(), bcrypt.gensalt()).decode(),
                'role': 'admin',
                'full_access': True
            }
            
            # Save updated config
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file)
                
    except FileNotFoundError:
        # Create default config
        config = create_default_config()
    
    # Ensure the pending_approvals key exists
    if 'pending_approvals' not in config:
        config['pending_approvals'] = {}
        with open('config.yaml', 'w') as file:
            yaml.dump(config, file)
    
    return config

def create_default_config():
    """Create a default configuration file with multiple admin users"""
    config = {
        'credentials': {
            'usernames': {
                # First admin (shorter credentials)
                'admin_123': {
                    'email': 'shivamatvit@gmail.com',
                    'name': 'Primary Admin',
                    'password': bcrypt.hashpw('aaddmin@123'.encode(), bcrypt.gensalt()).decode(),
                    'role': 'admin',
                    'full_access': True
                },
                # Second admin (longer, more complex credentials)
                'system_administrator_2023': {
                    'email': 'admin@dbbuddy.com',
                    'name': 'System Administrator',
                    'password': bcrypt.hashpw('ComplexP@$$w0rd2023!'.encode(), bcrypt.gensalt()).decode(),
                    'role': 'admin',
                    'full_access': True
                }
            }
        },
        'pending_approvals': {}
    }
    
    # Save config to file
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file)
    
    return config

def display_login_form(config):
    """Display login form and handle authentication with registration option"""
    st.title("DBBuddy: Your Smart Database Assistant")
    
    # Create tabs for login and registration
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        st.markdown("### Please log in to continue")
        
        # Simple login form
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if not username or not password:
                st.error("Please enter username and password")
                return False
            
            # Check credentials
            if username in config['credentials']['usernames']:
                stored_password = config['credentials']['usernames'][username]['password']
                # Verify password
                if bcrypt.checkpw(password.encode(), stored_password.encode()):
                    st.session_state.authentication_status = True
                    st.session_state.username = username
                    st.session_state.name = config['credentials']['usernames'][username]['name']
                    st.session_state.role = config['credentials']['usernames'][username].get('role', 'user')
                    st.session_state.full_access = config['credentials']['usernames'][username].get('full_access', False)
                    
                    # Save session state to file for persistence
                    try:
                        os.makedirs('.streamlit', exist_ok=True)
                        session_data = {
                            'username': username,
                            'name': st.session_state.name,
                            'role': st.session_state.role,
                            'full_access': st.session_state.full_access,
                            'auth_valid_until': time.time() + (24 * 60 * 60)  # 24 hour persistence
                        }
                        with open('.streamlit/session_state.json', 'w') as f:
                            json.dump(session_data, f)
                    except Exception as e:
                        # Continue even if saving fails
                        pass
                        
                    st.rerun()  # Refresh the page after successful login
                else:
                    st.error("Invalid password")
                    return False
            else:
                st.error("Username not found")
                return False
    
    with register_tab:
        st.markdown("### Register New Account")
        st.info("New users will have access to basic features only. Request full access after registration.")
        
        new_username = st.text_input("Username", key="register_username")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        new_name = st.text_input("Full Name", key="register_name")
        new_email = st.text_input("Email", key="register_email")
        
        if st.button("Register"):
            # Validate inputs
            if not new_username or not new_password or not confirm_password or not new_name or not new_email:
                st.error("Please fill all fields")
                return False
                
            if new_password != confirm_password:
                st.error("Passwords do not match")
                return False
                
            if new_username in config['credentials']['usernames']:
                st.error("Username already exists")
                return False
                
            # Create new user with limited access
            hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
            
            config['credentials']['usernames'][new_username] = {
                'email': new_email,
                'name': new_name,
                'password': hashed_password,
                'role': 'user',
                'full_access': False
            }
            
            # Save updated config
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file)
            
            st.success(f"Registration successful! You can now login with your credentials.")
            st.info("Note: You currently have limited access. Request full access from the dashboard.")
    
    return st.session_state.get('authentication_status', False)

def request_full_access(config):
    """Allow users to request full access to features"""
    if st.session_state.role != 'admin' and not st.session_state.full_access:
        with st.sidebar.expander("Request Full Access", expanded=False):
            st.write("Currently you have limited access to features.")
            st.write("Request full access to use Supabase connections and advanced visualizations.")
            
            if st.session_state.username in config.get('pending_approvals', {}):
                st.info("Your request is pending approval by the administrator.")
            else:
                if st.button("Request Full Access"):
                    # Add request to pending approvals
                    if 'pending_approvals' not in config:
                        config['pending_approvals'] = {}
                    
                    config['pending_approvals'][st.session_state.username] = {
                        'email': config['credentials']['usernames'][st.session_state.username]['email'],
                        'name': config['credentials']['usernames'][st.session_state.username]['name'],
                        'requested_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Save updated config
                    with open('config.yaml', 'w') as file:
                        yaml.dump(config, file)
                    
                    st.success("Access request submitted to admin shivamatvit@gmail.com")
                    st.info("You will be notified when your request is approved.")
                    st.rerun()

def admin_approval_panel(config):
    """Display admin panel for managing access requests"""
    if st.session_state.role == 'admin':
        with st.sidebar.expander("Admin Panel", expanded=False):
            st.markdown("### Pending Access Requests")
            
            if not config.get('pending_approvals'):
                st.info("No pending requests")
            else:
                for username, request_info in config.get('pending_approvals', {}).items():
                    st.markdown(f"**{request_info['name']}** ({username})")
                    st.markdown(f"Email: {request_info['email']}")
                    st.markdown(f"Requested: {request_info['requested_at']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Approve", key=f"approve_{username}"):
                            # Grant full access
                            config['credentials']['usernames'][username]['full_access'] = True
                            # Remove from pending approvals
                            del config['pending_approvals'][username]
                            # Save updated config
                            with open('config.yaml', 'w') as file:
                                yaml.dump(config, file)
                            st.success(f"Approved access for {username}")
                            st.rerun()
                    
                    with col2:
                        if st.button("Reject", key=f"reject_{username}"):
                            # Remove from pending approvals
                            del config['pending_approvals'][username]
                            # Save updated config
                            with open('config.yaml', 'w') as file:
                                yaml.dump(config, file)
                            st.success(f"Rejected access for {username}")
                            st.rerun()
                    
                    st.markdown("---")

def logout():
    """Log out the current user"""
    if st.sidebar.button("Logout"):
        st.session_state.authentication_status = None
        st.session_state.username = None
        st.session_state.name = None
        st.session_state.role = None
        st.session_state.full_access = False
        
        # Remove session file
        try:
            if os.path.exists('.streamlit/session_state.json'):
                os.remove('.streamlit/session_state.json')
        except:
            pass
            
        st.rerun()

def main():
    """Main function to run the application with role-based authentication"""
    # Initialize authentication
    config = initialize_authentication()
    
    # If not authenticated, display login form
    if 'authentication_status' not in st.session_state or st.session_state.authentication_status is not True:
        auth_status = display_login_form(config)
        # Exit if not authenticated
        if auth_status is not True:
            st.stop()
    
    # If we reach here, user is authenticated
    st.sidebar.title(f"Welcome, {st.session_state.name}")
    
    # Add logout button
    logout()
    
    # Show user management (only visible to admin)
    if st.session_state.role == 'admin':
        display_user_management(config)
        admin_approval_panel(config)
    
    # Show full access request option for regular users
    if st.session_state.role != 'admin':
        request_full_access(config)
    
    # Allow password reset
    allow_password_reset(config)
    
    # Display the main application content
    show_main_app()

def show_main_app():
    """Display the main application content after authentication with improved visualization tab"""
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

    # Database type selection based on access level
    if st.session_state.role == 'admin' or st.session_state.full_access:
        # Full access - show all database types
        new_db_type = st.radio("Choose your database type", ["SQLite", "Supabase"])
    else:
        # Limited access - only SQLite
        st.info("You currently have limited access. Only SQLite databases are available.")
        st.info("Request full access to use Supabase connections and advanced visualizations.")
        new_db_type = "SQLite"
    
    # Handle database type switching
    if st.session_state.current_db_type != new_db_type:
        st.session_state.db_initialized = False
        st.session_state.current_db_type = new_db_type
    
    if new_db_type == "SQLite":
        display_sqlite_connection()
    elif new_db_type == "Supabase" and (st.session_state.role == 'admin' or st.session_state.full_access):
        # Show setup instructions
        display_supabase_setup_instructions()
        # Show connection interface
        display_supabase_connection()
    
    # Display main content tabs only if database is initialized
    if st.session_state.db_initialized:
        # Create tabs for Chat and Visualization
        chat_tab, viz_tab = st.tabs(["Chat with Database", "Create Visualizations"])
        
        with chat_tab:
            has_full_access = st.session_state.role == 'admin' or st.session_state.full_access
            display_chat_interface(has_full_access)
        
        with viz_tab:
            # Display dedicated visualization interface
            display_visualization_interface()

def display_chat_interface(has_full_access=False):
    """Display the chat interface with RAG support, ensuring it always appears"""
    st.markdown("---")
    st.subheader("Chat with your Database")
    
    # Initialize chat history if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Create a stable container for all chat components
    chat_outer_container = st.container()
    
    with chat_outer_container:
        # Display chat history in a scrollable area
        chat_history_container = st.container()
        with chat_history_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Ensure the chat input is always visible by placing it in a separate container
        # This helps prevent UI disappearing issues
        input_container = st.container()
        
        with input_container:
            # Chat input with a unique key to prevent conflicts
            user_input = st.chat_input("Ask about your database...", key="db_chat_input")
            
            if user_input:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Show the user message immediately
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Generate response with a spinner
                with st.spinner("Thinking..."):
                    with st.chat_message("assistant"):
                        # Check if advanced visualization is requested but user doesn't have full access
                        viz_request = parse_visualization_request(user_input)
                        advanced_viz_requested = viz_request.get('requested', False) and viz_request.get('specific_viz') in ['correlation', 'heatmap', 'scatter']
                        
                        if advanced_viz_requested and not has_full_access:
                            restricted_response = "I'm sorry, but advanced visualizations like correlation plots, heatmaps, and scatter plots require full access. Please request full access from the sidebar."
                            st.markdown(restricted_response)
                            st.session_state.messages.append({"role": "assistant", "content": restricted_response})
                        else:
                            try:
                                # Generate response with the db data
                                response = generate_db_response(user_input, has_full_access)
                                if response:
                                    st.markdown(response)
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                else:
                                    error_msg = "I couldn't generate a response. Please try a different question."
                                    st.markdown(error_msg)
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            except Exception as e:
                                error_msg = f"Error generating response: {str(e)}"
                                st.markdown(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})

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

def detect_relationships(schema):
    """Detect potential relationships between tables by analyzing column names"""
    relationships = []
    common_id_patterns = ['id', '_id', 'key', 'code']
    
    # Get all tables and their columns
    tables = list(schema.keys())
    
    # Look for potential relationships based on column names
    for i, table1 in enumerate(tables):
        for j in range(i+1, len(tables)):
            table2 = tables[j]
            
            table1_cols = [col['name'].lower() for col in schema[table1]['columns']]
            table2_cols = [col['name'].lower() for col in schema[table2]['columns']]
            
            # Check for exact column name matches (potential foreign keys)
            for col1 in table1_cols:
                if col1 in table2_cols:
                    # Check if it looks like an ID column
                    if any(pattern in col1 for pattern in common_id_patterns):
                        relationships.append({
                            'table1': table1,
                            'table2': table2,
                            'column': col1,
                            'type': 'potential_fk'
                        })
            
            # Look for table name in column names (e.g., customer_id in orders table)
            table1_name = table1.lower().rstrip('s')  # Remove potential plural
            table2_name = table2.lower().rstrip('s')
            
            for col in table2_cols:
                if f"{table1_name}_id" in col or f"{table1_name}id" in col:
                    relationships.append({
                        'table1': table1,
                        'table2': table2,
                        'column': col,
                        'type': 'naming_convention'
                    })
            
            for col in table1_cols:
                if f"{table2_name}_id" in col or f"{table2_name}id" in col:
                    relationships.append({
                        'table1': table2,
                        'table2': table1,
                        'column': col,
                        'type': 'naming_convention'
                    })
    
    return relationships

@lru_cache(maxsize=8)
def create_enhanced_database_context(schema_hash):
    """Create a detailed context string with caching to reduce API usage"""
    # Convert the schema_hash back to the original schema
    # This function needs to be modified to work with the hash
    schema = st.session_state.schema
    
    context = []
    
    # Generate a more compact context
    # First, detect relationships between tables
    relationships = detect_relationships(schema)
    
    # Add database overview - keep it brief
    if relationships:
        context.append("Related tables:")
        for rel in relationships[:5]:  # Only include top 5 relationships
            context.append(f"- {rel['table1']} → {rel['table2']} via {rel['column']}")
        context.append("")
    
    # Process each table - limit the detail
    for table_name, table_info in list(schema.items())[:5]:  # Only process up to 5 tables
        # Basic table info
        table_desc = f"\nTable: {table_name} ({table_info['total_rows']} records)"
        
        # Column information - limit to essential columns
        important_cols = table_info['columns'][:10]  # First 10 columns only
        table_desc += "\nKey columns: " + ", ".join([f"{col['name']} ({col['type']})" for col in important_cols])
        
        # Very limited sample data
        if table_info['sample_data'] and len(table_info['sample_data']) > 0:
            table_desc += "\nExample row: " + str(table_info['sample_data'][0])
        
        context.append(table_desc)
    
    return "\n".join(context)

# When calling this function, use a simple hash
def get_schema_hash():
    """Create a simple hash of the schema for caching"""
    if 'schema' not in st.session_state:
        return "no_schema"
    
    # Get table names and row counts for a simple hash
    tables = []
    for table, info in st.session_state.schema.items():
        tables.append(f"{table}:{info['total_rows']}")
    
    return ",".join(sorted(tables))

def parse_visualization_request(prompt: str) -> dict:
    """Parse visualization requests from user prompts with enhanced reliability"""
    prompt_lower = prompt.lower()
    
    # Detect visualization request
    viz_request = {
        'requested': False,
        'type': None,
        'columns': [],
        'x_axis': None,
        'y_axis': None,
        'groupby': None,
        'specific_viz': None,
        'raw_prompt': prompt_lower  # Store the raw prompt for column extraction
    }
    
    # Clear pattern matching for visualization requests
    visualization_terms = [
        'visualize', 'visualization', 'chart', 'graph', 'plot', 'show me', 
        'display', 'create a', 'generate a', 'can you show', 'draw'
    ]
    
    # Check for general visualization request
    viz_request['requested'] = any(term in prompt_lower for term in visualization_terms)
    
    # Also check for specific visualization types
    specific_viz_terms = [
        'pie chart', 'bar chart', 'bar graph', 'line chart', 'line graph',
        'histogram', 'scatter plot', 'correlation', 'heatmap', 'box plot'
    ]
    
    if not viz_request['requested']:
        viz_request['requested'] = any(term in prompt_lower for term in specific_viz_terms)
    
    if not viz_request['requested']:
        return viz_request
    
    # Check visualization type with broad pattern matching
    if any(term in prompt_lower for term in ['pie', 'pie chart', 'pie graph']):
        viz_request['specific_viz'] = 'pie'
    elif any(term in prompt_lower for term in ['bar', 'bar chart', 'bar graph', 'bars']):
        viz_request['specific_viz'] = 'bar'
    elif any(term in prompt_lower for term in ['line', 'line chart', 'line graph', 'trend', 'over time']):
        viz_request['specific_viz'] = 'line'
    elif any(term in prompt_lower for term in ['scatter', 'scatter plot', 'scatterplot', 'points']):
        viz_request['specific_viz'] = 'scatter'
    elif any(term in prompt_lower for term in ['hist', 'histogram', 'distribution', 'frequency']):
        viz_request['specific_viz'] = 'histogram'
    elif any(term in prompt_lower for term in ['correlation', 'correlate', 'relationship between']):
        viz_request['specific_viz'] = 'correlation'
    elif any(term in prompt_lower for term in ['heatmap', 'heat map', 'heat']):
        viz_request['specific_viz'] = 'heatmap'
    elif any(term in prompt_lower for term in ['box', 'box plot', 'boxplot']):
        viz_request['specific_viz'] = 'box'
    
    # Try to extract axis information using flexible patterns
    import re
    
    # X-axis patterns
    x_axis_patterns = [
        r'x\s*axis\s*[=:]\s*([a-zA-Z_\s]+)',
        r'x-axis\s*[=:]\s*([a-zA-Z_\s]+)',
        r'x\s+should\s+be\s+([a-zA-Z_\s]+)',
        r'with\s+([a-zA-Z_\s]+)\s+on\s+the\s+x\s+axis',
        r'with\s+([a-zA-Z_\s]+)\s+as\s+x',
        r'x\s+is\s+([a-zA-Z_\s]+)',
        r'use\s+([a-zA-Z_\s]+)\s+for\s+x',
        r'use\s+([a-zA-Z_\s]+)\s+as\s+the\s+x',
        r'use\s+([a-zA-Z_\s]+)\s+on\s+the\s+x'
    ]
    
    # Y-axis patterns
    y_axis_patterns = [
        r'y\s*axis\s*[=:]\s*([a-zA-Z_\s]+)',
        r'y-axis\s*[=:]\s*([a-zA-Z_\s]+)',
        r'y\s+should\s+be\s+([a-zA-Z_\s]+)',
        r'with\s+([a-zA-Z_\s]+)\s+on\s+the\s+y\s+axis',
        r'with\s+([a-zA-Z_\s]+)\s+as\s+y',
        r'y\s+is\s+([a-zA-Z_\s]+)',
        r'use\s+([a-zA-Z_\s]+)\s+for\s+y',
        r'use\s+([a-zA-Z_\s]+)\s+as\s+the\s+y',
        r'use\s+([a-zA-Z_\s]+)\s+on\s+the\s+y'
    ]
    
    for pattern in x_axis_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            viz_request['x_axis'] = match.group(1).strip()
            break
    
    for pattern in y_axis_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            viz_request['y_axis'] = match.group(1).strip()
            break
    
    # Look for groupby information with flexible patterns
    group_patterns = [
        r'group\s+by\s+([a-zA-Z_\s]+)',
        r'grouped\s+by\s+([a-zA-Z_\s]+)',
        r'segmented\s+by\s+([a-zA-Z_\s]+)',
        r'categorized\s+by\s+([a-zA-Z_\s]+)',
        r'split\s+by\s+([a-zA-Z_\s]+)',
        r'separate\s+by\s+([a-zA-Z_\s]+)',
        r'using\s+([a-zA-Z_\s]+)\s+as\s+group',
        r'using\s+([a-zA-Z_\s]+)\s+for\s+grouping'
    ]
    
    for pattern in group_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            viz_request['groupby'] = match.group(1).strip()
            break
    
    return viz_request

def display_visualization_interface():
    """Display a dedicated interface for creating visualizations"""
    st.header("Create Visualizations")
    st.markdown("Select tables, columns, and visualization types to explore your data visually.")
    
    if 'schema' not in st.session_state or not st.session_state.schema:
        st.warning("Please connect to a database first.")
        return
    
    # Get connection for data queries
    connection = None
    if "db_engine" in st.session_state:
        connection = st.session_state.db_engine
    elif "supabase_client" in st.session_state:
        connection = st.session_state.supabase_client
    else:
        st.warning("No database connection found.")
        return
    
    # Create a separate container for visualization controls
    config_container = st.container()
    result_container = st.container()
    
    # Initialize session state for viz interface if not already done
    if 'viz_interface' not in st.session_state:
        st.session_state.viz_interface = {
            'selected_table': None,
            'table_data': None,
            'x_axis': None,
            'y_axis': None,
            'viz_type': 'bar',
            'group_by': None,
            'generated': False,
            'error': None
        }
    
    with config_container:
        # Use a form to prevent automatic reruns
        with st.form("visualization_form"):
            st.subheader("Configure Visualization")
            
            # 1. Table selection
            table_options = [table for table in st.session_state.schema.keys() 
                           if not table.startswith(('pg_', 'auth_', 'storage_'))]
            
            if not table_options:
                st.warning("No tables available for visualization.")
                st.form_submit_button("Submit", disabled=True)
                return
            
            # Default to the first table or previously selected table
            default_table_idx = 0
            if st.session_state.viz_interface['selected_table'] in table_options:
                default_table_idx = table_options.index(st.session_state.viz_interface['selected_table'])
            
            selected_table = st.selectbox(
                "Select Table", 
                options=table_options,
                index=default_table_idx,
                key="viz_table_select"
            )
            
            # Fetch fresh data for the selected table if needed
            if selected_table != st.session_state.viz_interface['selected_table'] or st.session_state.viz_interface['table_data'] is None:
                try:
                    # Execute simple query to get table data
                    query = f'SELECT * FROM "{selected_table}" LIMIT 1000'
                    columns, rows = execute_query(connection, query)
                    
                    if columns and rows:
                        df = pd.DataFrame(rows, columns=columns)
                        st.session_state.viz_interface['table_data'] = df
                        st.session_state.viz_interface['selected_table'] = selected_table
                    else:
                        st.warning(f"No data available in table {selected_table}")
                        st.session_state.viz_interface['table_data'] = None
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    st.session_state.viz_interface['table_data'] = None
            
            # Get current table data
            df = st.session_state.viz_interface['table_data']
            
            # If we have data, show configuration options
            if df is not None and not df.empty:
                # Get column types
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                
                # 2. Visualization Type
                viz_types = {
                    "bar": "Bar Chart",
                    "pie": "Pie Chart",
                    "line": "Line Chart",
                    "scatter": "Scatter Plot",
                    "histogram": "Histogram",
                    "box": "Box Plot"
                }
                
                selected_viz = st.selectbox(
                    "Visualization Type",
                    options=list(viz_types.keys()),
                    format_func=lambda x: viz_types[x],
                    index=list(viz_types.keys()).index(st.session_state.viz_interface['viz_type']) 
                        if st.session_state.viz_interface['viz_type'] in viz_types else 0
                )
                
                # 3. Columns selection based on visualization type
                col1, col2 = st.columns(2)
                
                with col1:
                    # X-axis selection
                    if selected_viz in ['bar', 'pie', 'box']:
                        x_options = categorical_cols
                        x_label = "Category Column (X axis)"
                    elif selected_viz in ['line', 'scatter']:
                        x_options = numeric_cols + date_cols
                        x_label = "X-axis (numeric/date)"
                    elif selected_viz == 'histogram':
                        x_options = numeric_cols
                        x_label = "Numeric Column"
                    else:
                        x_options = df.columns.tolist()
                        x_label = "X-axis Column"
                    
                    # Only show if we have options
                    if x_options:
                        # Find default index
                        default_x_idx = 0
                        if st.session_state.viz_interface['x_axis'] in x_options:
                            default_x_idx = x_options.index(st.session_state.viz_interface['x_axis'])
                        
                        x_axis = st.selectbox(
                            x_label,
                            options=x_options,
                            index=min(default_x_idx, len(x_options)-1)
                        )
                    else:
                        st.warning(f"No suitable columns for {x_label}")
                        x_axis = None
                
                with col2:
                    # Y-axis selection
                    if selected_viz in ['bar', 'line', 'scatter', 'box']:
                        y_options = numeric_cols
                        y_label = "Numeric Column (Y axis)"
                    elif selected_viz == 'pie':
                        y_options = numeric_cols if numeric_cols else ['count']
                        y_label = "Value Column (or count)"
                    elif selected_viz == 'histogram':
                        y_options = []  # Not needed for histogram
                        y_axis = None
                    else:
                        y_options = numeric_cols
                        y_label = "Y-axis Column"
                    
                    # Only show if we have options and need Y-axis
                    if y_options and selected_viz != 'histogram':
                        # Find default index
                        default_y_idx = 0
                        if st.session_state.viz_interface['y_axis'] in y_options:
                            default_y_idx = y_options.index(st.session_state.viz_interface['y_axis'])
                        
                        y_axis = st.selectbox(
                            y_label,
                            options=y_options,
                            index=min(default_y_idx, len(y_options)-1)
                        )
                    elif selected_viz == 'pie' and not numeric_cols:
                        st.info("Will use count for values")
                        y_axis = 'count'
                    elif selected_viz == 'histogram':
                        y_axis = None  # Not needed for histogram
                    else:
                        st.warning(f"No suitable columns for {y_label}")
                        y_axis = None
                
                # 4. Group by (optional)
                if selected_viz in ['bar', 'line', 'scatter'] and categorical_cols:
                    filtered_cats = [col for col in categorical_cols if col != x_axis]
                    
                    if filtered_cats:
                        group_options = ["None"] + filtered_cats
                        
                        # Default selection
                        default_group = 0
                        if st.session_state.viz_interface['group_by'] in filtered_cats:
                            default_group = filtered_cats.index(st.session_state.viz_interface['group_by']) + 1
                        
                        group_by = st.selectbox(
                            "Group by (optional)",
                            options=group_options,
                            index=min(default_group, len(group_options)-1)
                        )
                        
                        if group_by == "None":
                            group_by = None
                    else:
                        group_by = None
                else:
                    group_by = None
                
                # Check if we have all required data
                has_required_data = True
                if selected_viz in ['bar', 'line', 'scatter', 'box'] and (x_axis is None or y_axis is None):
                    has_required_data = False
                    st.warning("Both X and Y axes are required for this visualization type.")
                elif selected_viz in ['pie', 'histogram'] and x_axis is None:
                    has_required_data = False
                    st.warning("X axis is required for this visualization type.")
                
                # Store current selections
                st.session_state.viz_interface['x_axis'] = x_axis
                st.session_state.viz_interface['y_axis'] = y_axis
                st.session_state.viz_interface['viz_type'] = selected_viz
                st.session_state.viz_interface['group_by'] = group_by
                
                # Submit button
                submit_button = st.form_submit_button("Generate Visualization", disabled=not has_required_data)
                
                if submit_button and has_required_data:
                    st.session_state.viz_interface['generated'] = True
                    st.session_state.viz_interface['error'] = None
            else:
                st.warning("No data available to visualize.")
                st.form_submit_button("Submit", disabled=True)
    
    # Show visualization result in separate container
    with result_container:
        if st.session_state.viz_interface['generated']:
            try:
                st.subheader("Visualization Result")
                
                # Get current visualization configuration
                df = st.session_state.viz_interface['table_data']
                viz_type = st.session_state.viz_interface['viz_type']
                x_axis = st.session_state.viz_interface['x_axis']
                y_axis = st.session_state.viz_interface['y_axis']
                group_by = st.session_state.viz_interface['group_by']
                
                # Get column types again for clean code
                date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                
                # Standard figure size
                fig_width = 10
                fig_height = 6
                
                # Create appropriate chart
                if viz_type == 'bar' and x_axis:
                    create_bar_chart(df, x_axis, y_axis, group_by, fig_width, fig_height)
                elif viz_type == 'pie' and x_axis:
                    create_pie_chart(df, x_axis, y_axis, fig_width, fig_height)
                elif viz_type == 'line' and x_axis and y_axis:
                    create_line_chart(df, x_axis, y_axis, group_by, date_cols, fig_width, fig_height)
                elif viz_type == 'scatter' and x_axis and y_axis:
                    create_scatter_plot(df, x_axis, y_axis, group_by, fig_width, fig_height)
                elif viz_type == 'histogram' and x_axis:
                    create_histogram(df, x_axis, fig_width, fig_height)
                elif viz_type == 'box' and y_axis:
                    create_box_plot(df, x_axis, y_axis, fig_width, fig_height)
                else:
                    st.error("Could not determine how to create this visualization.")
                
                # Add a reset button
                if st.button("Create New Visualization"):
                    st.session_state.viz_interface['generated'] = False
                    st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.session_state.viz_interface['error'] = str(e)
                
                # Try to create a simple visualization as fallback
                try:
                    st.subheader("Simple Visualization (Fallback)")
                    df = st.session_state.viz_interface['table_data']
                    x_axis = st.session_state.viz_interface['x_axis']
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    if x_axis in df.columns:
                        df[x_axis].value_counts().head(10).plot(kind='bar', ax=ax)
                        plt.title(f"Counts of {x_axis}")
                    else:
                        # Just show any column we can
                        for col in df.columns:
                            if df[col].nunique() < 20:
                                df[col].value_counts().head(10).plot(kind='bar', ax=ax)
                                plt.title(f"Counts of {col}")
                                break
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except:
                    st.warning("Could not create any visualization. Please try different settings.")

def create_bar_chart(df, x_axis, y_axis, groupby_col, fig_width, fig_height):
    """Create a bar chart visualization with reliable rendering"""
    try:
        st.markdown(f"#### Bar Chart: {y_axis} by {x_axis}")
        
        # Create figure explicitly with exact size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        plt.clf()  # Clear any existing plots
        ax = plt.gca()  # Get current axis
        
        if groupby_col:
            # Handle grouped data
            try:
                # Create pivot table for grouped bar chart
                pivot_df = df.pivot_table(
                    index=x_axis,
                    columns=groupby_col,
                    values=y_axis,
                    aggfunc='sum'
                )
                
                # Limit to top 15 categories if there are too many
                if len(pivot_df) > 15:
                    pivot_df = pivot_df.sort_values(pivot_df.columns[0], ascending=False).head(15)
                
                # Plot the pivot table as grouped bars
                pivot_df.plot(kind='bar', ax=ax, width=0.8)
                plt.title(f"{y_axis} by {x_axis} grouped by {groupby_col}", fontsize=16)
                plt.ylabel(y_axis, fontsize=14)
                plt.legend(title=groupby_col)
            except Exception as e:
                st.warning(f"Could not create grouped bar chart: {str(e)}. Creating simple chart instead.")
                # Fallback to a simpler approach with a single group
                plot_top = df.groupby(x_axis)[y_axis].sum().sort_values(ascending=False).head(15)
                plot_top.plot(kind='bar', ax=ax, color='skyblue')
                plt.title(f"{y_axis} by {x_axis}", fontsize=16)
                plt.ylabel(y_axis, fontsize=14)
        else:
            # Simple bar chart
            try:
                # Group, sum, and take top 15 categories
                plot_df = df.groupby(x_axis)[y_axis].sum().sort_values(ascending=False).head(15)
                plot_df.plot(kind='bar', ax=ax, color='skyblue')
                plt.title(f"{y_axis} by {x_axis}", fontsize=16)
                plt.ylabel(y_axis, fontsize=14)
            except Exception as e:
                st.warning(f"Error in simple bar chart: {str(e)}. Trying basic count.")
                # Super simple fallback to just counts
                df[x_axis].value_counts().head(15).plot(kind='bar', ax=ax, color='skyblue')
                plt.title(f"Count by {x_axis}", fontsize=16)
                plt.ylabel("Count", fontsize=14)
        
        # Common formatting
        plt.xlabel(x_axis, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()  # Ensure labels fit in the figure
        
        # Display the plot
        st.pyplot(fig)
        
        # Always close the figure to prevent matplotlib memory issues
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")
        # Last resort - try with a very basic approach
        try:
            # Create a new clean figure
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 5))
            df[x_axis].value_counts().head(10).plot.bar(ax=ax)
            plt.title(f"Count of {x_axis}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.error("Could not create any bar chart visualization.")

def create_pie_chart(df, x_axis, y_axis, fig_width, fig_height):
    """Create a pie chart visualization"""
    try:
        st.markdown(f"#### Pie Chart: Distribution of {x_axis}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        if y_axis and y_axis != 'count':
            # Use the specified numeric column as values
            grouped_data = df.groupby(x_axis)[y_axis].sum().sort_values(ascending=False)
            title = f"{y_axis} by {x_axis}"
        else:
            # Use counts as values
            grouped_data = df[x_axis].value_counts()
            title = f"Distribution of {x_axis}"
        
        # Limit slices to prevent overloading
        if len(grouped_data) > 8:
            # Keep top 7 slices and group the rest as "Other"
            other_sum = grouped_data[7:].sum()
            grouped_data = grouped_data.head(7)
            grouped_data["Other"] = other_sum
        
        # Create pie chart
        patches, texts, autotexts = ax.pie(
            grouped_data,
            labels=grouped_data.index,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False,
            textprops={'fontsize': 12}
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"Error creating pie chart: {str(e)}")
        # Try a simple fallback
        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 5))
            # Simple value counts pie chart
            df[x_axis].value_counts().head(5).plot.pie(ax=ax, autopct='%1.1f%%')
            plt.title(f"Top values of {x_axis}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.error("Could not create any pie chart visualization.")

def create_line_chart(df, x_axis, y_axis, groupby_col, date_cols, fig_width, fig_height):
    """Create a line chart visualization with proper indentation"""
    try:
        st.markdown(f"#### Line Chart: {y_axis} vs {x_axis}")
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Process data differently if x_axis is a date
        is_date = x_axis in date_cols
        
        if groupby_col:
            # For each group, create a line
            for group_name, group_df in df.groupby(groupby_col):
                # Sort by x-axis
                plot_df = group_df.sort_values(by=x_axis).copy()
                # Plot the line with a label for the legend
                ax.plot(plot_df[x_axis], plot_df[y_axis], marker='o', label=str(group_name))
                
            # Add a legend
            ax.legend(title=groupby_col, loc='best')
            plt.title(f"{y_axis} vs {x_axis} by {groupby_col}", fontsize=16)
        else:
            # Simple line chart - sort by x-axis
            plot_df = df.sort_values(by=x_axis).copy()
            # Plot the line
            ax.plot(plot_df[x_axis], plot_df[y_axis], marker='o', color='steelblue')
            plt.title(f"{y_axis} vs {x_axis}", fontsize=16)
        
        # Common formatting
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel(y_axis, fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Handle date formatting
        if is_date:
            plt.xticks(rotation=45)
            fig.autofmt_xdate()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error creating line chart: {str(e)}")
        # Try a simpler version
        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Group by x-axis and take mean of y-axis
            plot_data = df.groupby(x_axis)[y_axis].mean().sort_index()
            plot_data.plot(kind='line', marker='o', ax=ax)
            
            plt.title(f"{y_axis} by {x_axis}")
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.error("Could not create any line chart visualization.")

def create_histogram(df, x_axis, fig_width, fig_height):
    """Create a histogram visualization"""
    try:
        st.markdown(f"#### Histogram: Distribution of {x_axis}")
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create the histogram with KDE (density curve)
        sns.histplot(data=df, x=x_axis, kde=True, ax=ax)
        
        # Add formatting
        plt.title(f"Distribution of {x_axis}", fontsize=16)
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error creating histogram: {str(e)}")
        # Try a simpler approach
        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 5))
            df[x_axis].plot.hist(ax=ax, bins=20)
            plt.title(f"Distribution of {x_axis}")
            plt.xlabel(x_axis)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.error("Could not create histogram visualization.")

def create_scatter_plot(df, x_axis, y_axis, groupby_col, fig_width, fig_height):
    """Create a scatter plot visualization"""
    try:
        st.markdown(f"#### Scatter Plot: {y_axis} vs {x_axis}")
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        if groupby_col:
            # Use seaborn's scatterplot with hue for grouping
            scatter = sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=groupby_col, ax=ax)
            
            # Try to place the legend outside if there are many categories
            if df[groupby_col].nunique() > 5:
                plt.legend(title=groupby_col, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.legend(title=groupby_col)
                
            plt.title(f"{y_axis} vs {x_axis} by {groupby_col}", fontsize=16)
        else:
            # Simple scatter plot
            scatter = sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
            plt.title(f"{y_axis} vs {x_axis}", fontsize=16)
        
        # Common formatting
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel(y_axis, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error creating scatter plot: {str(e)}")
        # Try a simpler approach
        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.scatter(df[x_axis], df[y_axis], alpha=0.7)
            plt.title(f"{y_axis} vs {x_axis}")
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.error("Could not create scatter plot visualization.")

def create_correlation_heatmap(df, numeric_cols, has_full_access):
    """Create a correlation heatmap visualization"""
    if not has_full_access:
        st.warning("Correlation heatmaps require full access")
        return
    
    try:
        st.markdown("#### Correlation Heatmap")
        
        # Check if we have enough numeric columns
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")
            return
            
        # Calculate correlation matrix
        corr = df[numeric_cols].corr()
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap with better formatting
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1, vmax=1,
            linewidths=0.5,
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        plt.title("Correlation Matrix", fontsize=16)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        # Try simpler version
        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 8))
            # Use only a subset of columns if there are too many
            if len(numeric_cols) > 8:
                use_cols = numeric_cols[:8]
            else:
                use_cols = numeric_cols
                
            # Simple correlation heatmap
            sns.heatmap(df[use_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.error("Could not create correlation heatmap.")

def create_box_plot(df, x_axis, y_axis, fig_width, fig_height):
    """Create a box plot visualization"""
    try:
        st.markdown(f"#### Box Plot: Distribution of {y_axis}")
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        if x_axis:
            # Check if x_axis has too many unique values
            if df[x_axis].nunique() > 12:
                st.warning(f"Too many categories in {x_axis}. Showing top 12 by frequency.")
                # Get top categories by frequency
                top_cats = df[x_axis].value_counts().head(12).index.tolist()
                # Filter data to top categories
                plot_df = df[df[x_axis].isin(top_cats)]
            else:
                plot_df = df
            
            # Create box plot grouped by x_axis
            sns.boxplot(data=plot_df, x=x_axis, y=y_axis, ax=ax)
            plt.title(f"Distribution of {y_axis} by {x_axis}", fontsize=16)
            plt.xlabel(x_axis, fontsize=14)
            # Rotate x-axis labels if many categories
            plt.xticks(rotation=45)
        else:
            # Simple box plot of y_axis only
            sns.boxplot(data=df, y=y_axis, ax=ax)
            plt.title(f"Distribution of {y_axis}", fontsize=16)
        
        plt.ylabel(y_axis, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Error creating box plot: {str(e)}")
        # Try simpler version
        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 6))
            # Simple boxplot
            df.boxplot(column=y_axis, ax=ax)
            plt.title(f"Distribution of {y_axis}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except:
            st.error("Could not create box plot visualization.")

def display_supabase_connection():
    """Display Supabase connection interface with improved direct connection"""
    st.info("""
    ### Easy Supabase Connection
    
    Connect to any Supabase database by providing:
    
    1. **Supabase URL**: Your project URL from the Supabase dashboard
    2. **API Key**: Your anon/public key (found in API settings)
    3. **Connection Method**: Choose REST API (easier) or PostgreSQL Direct
    
    No special setup required - just connect and start querying!
    """)
    
    # Connection inputs
    supabase_url = st.text_input(
        "Supabase URL", 
        placeholder="https://your-project-id.supabase.co",
        help="Found in your Supabase project settings"
    )
    
    anon_key = st.text_input(
        "API Key (anon/public)",
        type="password",
        help="Use your project's anon key for public access"
    )
    
    # Connection type selector
    connection_type = st.radio(
        "Connection Method",
        ["REST API (Recommended)", "PostgreSQL Direct"],
        help="REST API is simpler but PostgreSQL Direct offers more capabilities"
    )
    
    if st.button("Connect to Supabase"):
        if not supabase_url or not anon_key:
            st.error("Please provide both Supabase URL and API Key")
            return
        
        with st.spinner("Connecting to Supabase..."):
            # Use our improved connection function
            status, result = initialize_supabase_connection(supabase_url, anon_key, connection_type)
            
            if status == "success":
                st.success(f"Successfully connected to Supabase!")
                
                # Show database info
                if isinstance(result, dict) and len(result) > 0:
                    st.write(f"Found {len(result)} tables:")
                    for table_name, info in list(result.items())[:5]:
                        row_count = info.get('total_rows', 0)
                        st.write(f"- {table_name}: {row_count} records")
                    
                    if len(result) > 5:
                        st.write(f"... and {len(result) - 5} more tables")
                else:
                    st.warning("Connected successfully but no tables were found.")
            else:
                st.error(result)
                st.error("If you're using PostgreSQL Direct, ensure your IP is allowed in Supabase Project Settings > Database > Network Access")
                st.session_state.db_initialized = False

def extract_project_ref(url):
    """Extract project reference from Supabase URL"""
    import re
    match = re.search(r'https://([^.]+)\.supabase\.co', url)
    if match:
        return match.group(1)
    return None

def extract_schema_from_rest(client):
    """Extract database schema using REST API calls without RPC functions"""
    schema = {}
    
    try:
        # First try getting tables via information schema directly with proper query
        tables_data = []
        
        try:
            response = client.rpc('get_schema_info').execute()
            if hasattr(response, 'data') and response.data:
                tables_data = [{'name': item['table']} for item in response.data 
                              if item['schema'] == 'public' and not item['table'].startswith(('pg_', 'auth_', 'storage_'))]
        except Exception:
            # Try alternative approach using direct SQL query if RPC fails
            try:
                # Try direct query to information_schema
                response = client.table('information_schema.tables').select('table_name') \
                    .eq('table_schema', 'public').execute()
                
                if hasattr(response, 'data') and response.data:
                    tables_data = [{'name': item['table_name']} for item in response.data 
                                 if not item['table_name'].startswith(('pg_', 'auth_', 'storage_'))]
            except Exception:
                # Last resort: try common table names
                for common_table in ['users', 'customers', 'products', 'orders', 'employees', 'items', 'sales']:
                    try:
                        test = client.table(common_table).select('count').limit(1).execute()
                        if hasattr(test, 'data'):
                            tables_data.append({'name': common_table})
                    except:
                        pass
        
        # If still no tables found, add a placeholder to prevent errors
        if not tables_data:
            st.warning("No tables found or accessible in this Supabase project.")
            schema["sample_table"] = {
                'columns': [
                    {'name': 'id', 'type': 'integer', 'nullable': False},
                    {'name': 'name', 'type': 'text', 'nullable': True},
                    {'name': 'created_at', 'type': 'timestamp', 'nullable': True}
                ],
                'total_rows': 0,
                'sample_data': [{'id': 1, 'name': 'Sample', 'created_at': '2023-01-01'}],
                'is_sample': True
            }
            return schema
        
        # Process each table with better error handling
        for table_data in tables_data:
            table_name = table_data.get('name')
            
            if not table_name or table_name in ('error_info',):
                continue
                
            try:
                # Get sample data to infer column types with proper error handling
                try:
                    sample_response = client.table(table_name).select('*').limit(5).execute()
                    sample_data = sample_response.data if hasattr(sample_response, 'data') else []
                except Exception:
                    sample_data = []
                
                # Count records with fallback options
                row_count = 0
                try:
                    count_response = client.table(table_name).select('*', count='exact').limit(1).execute()
                    row_count = count_response.count if hasattr(count_response, 'count') else 0
                except:
                    row_count = len(sample_data) if sample_data else 0
                
                # Infer column info from sample data
                columns_info = []
                if sample_data and len(sample_data) > 0:
                    for col_name, col_value in sample_data[0].items():
                        col_type = infer_column_type(col_value)
                        columns_info.append({
                            'name': col_name,
                            'type': col_type,
                            'nullable': True  # Default assumption
                        })
                
                # Store table information
                schema[table_name] = {
                    'columns': columns_info,
                    'total_rows': row_count,
                    'sample_data': sample_data
                }
            except Exception as table_error:
                # Create minimal table schema to prevent downstream errors
                schema[table_name] = {
                    'columns': [{'name': 'id', 'type': 'integer', 'nullable': False}],
                    'total_rows': 0,
                    'sample_data': [],
                    'processing_error': str(table_error)
                }
        
        return schema
        
    except Exception as e:
        st.error(f"Error fetching schema: {str(e)}")
        # Return a minimal schema to prevent downstream errors
        return {
            "sample_table": {
                'columns': [
                    {'name': 'id', 'type': 'integer', 'nullable': False},
                    {'name': 'name', 'type': 'text', 'nullable': True},
                    {'name': 'created_at', 'type': 'timestamp', 'nullable': True}
                ],
                'total_rows': 0,
                'sample_data': [{'id': 1, 'name': 'Sample', 'created_at': '2023-01-01'}],
                'is_sample': True
            }
        }

def infer_column_type(value):
    """Infer SQL column type from Python value"""
    if value is None:
        return "unknown"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "double precision" 
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, (list, dict)):
        return "json"
    else:
        # Check if it's a date/timestamp string
        if isinstance(value, str):
            import re
            # Simple date format check
            if re.match(r'^\d{4}-\d{2}-\d{2}', value):
                return "timestamp"
        return "text"

def get_direct_db_schema(engine):
    """Get schema using direct database connection"""
    inspector = inspect(engine)
    schema = {}
    
    try:
        # Get list of all tables
        tables = inspector.get_table_names()
        
        for table_name in tables:
            # Skip system tables
            if table_name.startswith(('pg_', 'auth_', 'storage_')):
                continue
                
            # Get column information
            column_info = []
            for col in inspector.get_columns(table_name):
                column_info.append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True)
                })
            
            # Get sample data and row count
            with engine.connect() as connection:
                # Get total row count
                count_query = text(f'SELECT COUNT(*) FROM "{table_name}"')
                total_rows = connection.execute(count_query).scalar()
                
                # Get sample data
                sample_query = text(f'SELECT * FROM "{table_name}" LIMIT 5')
                result = connection.execute(sample_query)
                
                # Convert result to list of dictionaries
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
       
    3. **Or use a sample database**:
       - Click "Create Sample Cafe Database" to use a pre-populated database
    
    Note: Your database will be temporarily stored and removed when you close the application.
    """)
    
    # File uploader for existing database
    uploaded_file = st.file_uploader("Upload SQLite Database", type=['db', 'sqlite'])
    
    # Create new database option
    col1, col2 = st.columns(2)
    
    with col1:
        new_db_name = st.text_input("Or create a new database", placeholder="Enter database name (e.g., my_database.db)")
        create_new = st.button("Create New Database")
    
    with col2:
        st.write("Use a sample database with pre-populated data:")
        create_sample = st.button("Create Sample Cafe Database")
    
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

    elif create_sample:
        create_and_connect_to_sample_db()

def generate_db_response(prompt: str, has_full_access=True):
    """Main function to generate the database response with refined visualizations and error handling"""
    try:
        # Check if database is initialized
        if 'schema' not in st.session_state or not st.session_state.schema:
            return "Please connect to a database first."
            
        # Make sure we have a connection
        connection = None
        if "db_engine" in st.session_state:
            connection = st.session_state.db_engine 
        elif "supabase_client" in st.session_state:
            connection = st.session_state.supabase_client
        else:
            return "No database connection found. Please connect to a database first."
            
        # Parse visualization request from the prompt
        viz_request = parse_visualization_request(prompt)
        
        # Analyze query intent
        intent = analyze_query_intent(prompt)
        
        # Build and execute the dynamic query with better error handling
        try:
            query, table = build_dynamic_query(intent, st.session_state.schema)
            if not query:
                return "Could not determine how to query the database. Please try a different question."
            
            # Execute the query
            result = execute_query(connection, query)
            df = format_query_result(result, query)
        except Exception as query_error:
            st.error(f"Query execution error: {str(query_error)}")
            return f"I couldn't execute the query. Error: {str(query_error)}"
        
        # Check if we have valid results
        if isinstance(df, str):
            return df
        
        if isinstance(df, pd.DataFrame):
            # Check if this is an error DataFrame
            if 'Error' in df.columns and len(df.columns) == 1:
                return f"Database query error: {df['Error'].iloc[0]}"
                
            if df.empty:
                return "The query returned no results. Try a different question or table."
                
            # Display the dataframe results
            st.markdown("### Query Results")
            st.dataframe(df)
            
            # For visualization requests, direct users to the Visualization tab
            if viz_request.get('requested', False):
                viz_tab_message = """
To create visualizations for this data, please:
1. Go to the "Create Visualizations" tab above
2. Select the visualization type and appropriate columns
3. Click "Generate Visualization" to see the results

The visualization interface provides more options and better control for exploring your data visually.
                """
                st.info(viz_tab_message)
            
            # Generate analysis and response
            try:
                # Perform basic analysis on the results
                analysis = analyze_results(df, intent)
                
                # Generate a refined response
                schema_hash = get_schema_hash()
                enhanced_context = create_enhanced_database_context(schema_hash)
                refined_text = refine_chatgpt_response_enhanced(prompt, df, analysis, query, enhanced_context)
                
                # Combine the refined text into the final response
                final_response = f"**Analysis:**\n\n{refined_text}"
                return final_response
            except Exception as analysis_error:
                st.error(f"Error generating analysis: {str(analysis_error)}")
                # Provide a simple response based on the data
                return f"Query complete. Found {len(df)} results from the {table} table."
        else:
            return "The query did not return valid data."
    
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return f"I encountered an error while processing your query: {str(e)}"

def refine_chatgpt_response_enhanced(prompt: str, df: pd.DataFrame, analysis: dict, query: str, enhanced_context: str) -> str:
    """Generate a refined response with improved visualization handling"""
    try:
        # Check if this is a visualization request
        viz_request = parse_visualization_request(prompt)
        
        # If visualization is requested, provide guidance to use the visualization tab
        if viz_request.get('requested', False):
            # Create guidance for using the visualization tab
            viz_guidance = """
To create visualizations from this data, please use the 'Create Visualizations' tab where you can:
1. Select the visualization type (bar chart, line chart, etc.)
2. Choose appropriate columns for X and Y axes
3. Add grouping if needed
4. Generate interactive visualizations

The visualization interface provides more options and better control than I can offer directly.
            """
            
            # Include this guidance with the regular analysis
            context = f"""Data summary: {len(df)} records with columns: {', '.join(df.columns[:10])}

Sample data: {df.head(3).to_string(index=False) if not df.empty else 'No data'}

{enhanced_context[:300]}...

Include that this is a visualization request and direct the user to the Visualization tab.
"""
            
            # Generate response with guidance
            regular_response = generate_response(prompt, db_context=context)
            
            # Ensure visualization guidance is included
            return f"{regular_response}\n\n{viz_guidance}"
        
        # For non-visualization requests, proceed as normal
        data_sample = []
        if not df.empty:
            # Only include up to 5 rows and summarize the rest
            sample_size = min(5, len(df))
            data_sample = df.head(sample_size).to_dict(orient='records')
            if len(df) > sample_size:
                data_sample.append({"note": f"...and {len(df) - sample_size} more rows (truncated)"})
        else:
            data_sample = "No data available"
        
        # Create context for the response
        context = f"""Based on database results with {len(df) if not df.empty else 0} records:
        
Sample: {data_sample}

Key info: 
- Columns: {', '.join(df.columns[:10]) if not df.empty else 'None'}
- {enhanced_context[:500] + '...' if len(enhanced_context) > 500 else enhanced_context}

Provide a concise, friendly response with insights. Use formatting when helpful.
"""

        # Use more efficient model and limit tokens
        refined_response = generate_response(prompt, db_context=context)
        return refined_response
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

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
    """Build a robust SQL query that works across all database types"""
    # Handle empty schema or error cases
    if not schema:
        return None, "No tables found in the schema"
    
    # Skip internal tables that might cause issues
    filtered_schema = {k: v for k, v in schema.items() 
                      if k not in ('error_info', '_metadata', 'extensions', 'information_schema')}
    
    if not filtered_schema:
        return None, "No usable tables found in the schema"
    
    # Find relevant table with improved matching
    relevant_table = find_relevant_table(filtered_schema, intent.get('fields_of_interest', []))
    if not relevant_table:
        # Fallback: use first table
        for table_name in filtered_schema:
            if not table_name.startswith(('pg_', 'auth_', 'storage_')):
                relevant_table = table_name
                break
    
    if not relevant_table:
        return None, "No suitable tables found in database"
    
    # Ensure table exists and has structure
    if relevant_table not in filtered_schema:
        return None, f"Table {relevant_table} not found in schema"
    
    table_info = filtered_schema[relevant_table]
    
    # Get available columns with safeguards
    if 'columns' not in table_info or not table_info.get('columns'):
        # Fallback to simple SELECT * query
        return f'SELECT * FROM "{relevant_table}" LIMIT 50', relevant_table
    
    # Get safe column names (prevent SQL injection)
    available_columns = [f'"{col["name"]}"' for col in table_info['columns']]
    if not available_columns:
        # If no columns are available, use SELECT *
        return f'SELECT * FROM "{relevant_table}" LIMIT 50', relevant_table
    
    # Build a safe, universally compatible query
    query = f'SELECT {", ".join(available_columns)} FROM "{relevant_table}"'
    
    # Add simple WHERE clause if we have specific fields of interest
    if intent.get('fields_of_interest') and available_columns:
        # Try to find matching columns
        column_matches = []
        for field in intent.get('fields_of_interest', []):
            for col in table_info['columns']:
                if field.lower() in col['name'].lower():
                    column_matches.append(col['name'])
        
        # Add WHERE clause for string-type columns
        if column_matches:
            conditions = []
            for col_name in column_matches:
                # Check if it's likely a string column
                col_info = next((col for col in table_info['columns'] if col['name'] == col_name), None)
                if col_info and ('char' in str(col_info['type']).lower() or 'text' in str(col_info['type']).lower()):
                    conditions.append(f'"{col_name}" IS NOT NULL')
            
            if conditions:
                query += f" WHERE {' AND '.join(conditions)}"
    
    # Add ORDER BY for numeric columns if we're comparing values
    if intent.get('comparison'):
        # Find numeric columns
        numeric_cols = []
        for col in table_info['columns']:
            col_type = str(col['type']).lower()
            if any(num_type in col_type for num_type in ['int', 'float', 'double', 'decimal', 'numeric']):
                numeric_cols.append(col['name'])
        
        if numeric_cols:
            # Order by the first numeric column DESC for comparisons
            query += f' ORDER BY "{numeric_cols[0]}" DESC'
    
    # Add safe LIMIT
    query += " LIMIT 50"
    
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

def display_supabase_setup_instructions():
    with st.expander("First-time Supabase Setup"):
        st.markdown("""
        ### One-time Supabase Setup
        
        Before connecting to Supabase, you need to run these SQL functions in your Supabase SQL Editor:
        
        1. Go to your Supabase project
        2. Click on "SQL Editor" in the left sidebar
        3. Create a "New Query"
        4. Copy and paste the SQL below
        5. Click "Run"
        
        This will create the necessary helper functions to extract database schema and run queries.
        """)
        
        st.code("""
-- Function to get schema information
CREATE OR REPLACE FUNCTION get_schema_info()
RETURNS TABLE(schema text, "table" text) 
LANGUAGE plpgsql SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY 
  SELECT table_schema::text, table_name::text
  FROM information_schema.tables
  WHERE table_schema NOT IN ('pg_catalog', 'information_schema');
END;
$$;

-- Function to get table columns
CREATE OR REPLACE FUNCTION get_table_columns(target_table text)
RETURNS TABLE(column_name text, data_type text, is_nullable text) 
LANGUAGE plpgsql SECURITY DEFINER
AS $$
BEGIN
  RETURN QUERY 
  SELECT column_name::text, data_type::text, is_nullable::text
  FROM information_schema.columns
  WHERE table_schema = 'public' AND table_name = target_table;
END;
$$;

-- Function to get row count
CREATE OR REPLACE FUNCTION get_row_count(target_table text)
RETURNS TABLE(count bigint) 
LANGUAGE plpgsql SECURITY DEFINER
AS $$
DECLARE
  query text;
BEGIN
  query := 'SELECT COUNT(*) FROM public.' || quote_ident(target_table);
  RETURN QUERY EXECUTE query;
END;
$$;

-- Function to run arbitrary queries safely
CREATE OR REPLACE FUNCTION run_query(query_text text)
RETURNS SETOF json 
LANGUAGE plpgsql SECURITY DEFINER
AS $$
BEGIN
  -- Convert results to JSON to handle different data types including UUIDs
  RETURN QUERY EXECUTE 
    'WITH query_result AS (' || query_text || ') 
     SELECT row_to_json(query_result) FROM query_result';
END;
$$;
        """, language="sql")
        
        st.info("After running these SQL functions, you'll be able to connect to your Supabase database without any errors.")

def initialize_supabase_connection(url, api_key, connection_type):
    """Initialize Supabase connection with improved error handling"""
    try:
        if connection_type == "REST API (Recommended)":
            # Create Supabase client
            client = create_client(url, api_key)
            
            # Test the connection with a simple query
            try:
                # Try using rpc method first
                test_response = client.rpc('get_schema_info').execute()
            except Exception:
                # Fallback to direct table access if RPC fails
                try:
                    test_response = client.table('information_schema.tables').select('table_name').limit(1).execute()
                except Exception as e:
                    # If both methods fail, show specific error
                    return "error", f"Connection to Supabase failed: {str(e)}. Please check your credentials and ensure you've executed the setup SQL functions."
            
            # Get database schema through REST API
            schema = extract_schema_from_rest(client)
            
            # Store in session state
            st.session_state.supabase_client = client
            st.session_state.schema = schema
            st.session_state.db_initialized = True
            st.session_state.connection_type = "rest"
            st.session_state.current_db_type = "Supabase"
            
            return "success", schema
            
        else:  # PostgreSQL Direct
            # Extract project ref from URL
            project_ref = extract_project_ref(url)
            if not project_ref:
                return "error", "Invalid Supabase URL format. Expected format: https://project-id.supabase.co"
            
            # Build connection string with error handling for special characters
            try:
                connection_string = f"postgresql://postgres:{api_key}@db.{project_ref}.supabase.co:5432/postgres"
                
                # Create SQLAlchemy engine with connection pooling disabled to avoid timeouts
                engine = create_engine(connection_string, poolclass=NullPool)
                
                # Verify connection with a simple query
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                # Get schema with error handling
                schema = get_direct_db_schema(engine)
                
                # Store in session state
                st.session_state.db_engine = engine
                st.session_state.schema = schema
                st.session_state.db_initialized = True
                st.session_state.connection_type = "postgresql"
                st.session_state.current_db_type = "Supabase"
                
                return "success", schema
            except Exception as pg_error:
                return "error", f"PostgreSQL connection error: {str(pg_error)}. Check your credentials and ensure your IP is allowed in Supabase network settings."
            
    except Exception as e:
        # Return detailed error message
        return "error", f"Connection error: {str(e)}"

def display_user_management(config):
    """Display user management options for admin users"""
    st.sidebar.title("User Management")
    
    # List of admin usernames that should be protected
    admin_usernames = ['admin_123', 'system_administrator_2023']
    
    # Only show user management to admin users
    if st.session_state.role == 'admin':
        with st.sidebar.expander("User Management", expanded=False):
            # Add new user form
            st.markdown("### Add New User")
            new_username = st.text_input("Username", key="new_username")
            new_name = st.text_input("Full Name", key="new_name")
            new_email = st.text_input("Email", key="new_email")
            new_password = st.text_input("Password", type="password", key="new_password")
            full_access = st.checkbox("Grant Full Access", value=False, key="grant_access")
            
            if st.button("Add User"):
                if not new_username or not new_name or not new_email or not new_password:
                    st.error("Please fill all fields")
                    return
                    
                if new_username in config['credentials']['usernames']:
                    st.error("Username already exists")
                    return
                
                # Hash the password
                hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                
                # Add user to config
                config['credentials']['usernames'][new_username] = {
                    'email': new_email,
                    'name': new_name,
                    'password': hashed_password,
                    'role': 'user',
                    'full_access': full_access
                }
                
                # Save updated config
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file)
                
                st.success(f"User {new_username} added successfully")
                st.rerun()
            
            # Display existing users
            st.markdown("### Existing Users")
            for username, user_data in config['credentials']['usernames'].items():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{username}** ({user_data['name']})")
                    st.write(f"Role: {user_data.get('role', 'user')}")
                    st.write(f"Access: {'Full' if user_data.get('full_access', False) else 'Limited'}")
                
                with col2:
                    if username not in admin_usernames:  # Prevent modifying admins
                        if st.button("Toggle Access", key=f"toggle_{username}"):
                            # Toggle full access
                            config['credentials']['usernames'][username]['full_access'] = not config['credentials']['usernames'][username].get('full_access', False)
                            # Save updated config
                            with open('config.yaml', 'w') as file:
                                yaml.dump(config, file)
                            st.rerun()
                
                with col3:
                    if username not in admin_usernames:  # Prevent deleting admins
                        if st.button("Delete", key=f"del_{username}"):
                            # Remove user from config
                            del config['credentials']['usernames'][username]
                            # Save updated config
                            with open('config.yaml', 'w') as file:
                                yaml.dump(config, file)
                            st.rerun()
                
                st.markdown("---")

def allow_password_reset(config):
    """Allow users to reset their password"""
    with st.sidebar.expander("Reset Password", expanded=False):
        current_password = st.text_input("Current Password", type="password", key="current_pw")
        new_password = st.text_input("New Password", type="password", key="new_pw")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pw")
        
        if st.button("Reset Password"):
            if not current_password or not new_password or not confirm_password:
                st.error("Please fill all fields")
                return
                
            if new_password != confirm_password:
                st.error("New passwords do not match")
                return
                
            # Verify current password
            stored_password = config['credentials']['usernames'][st.session_state.username]['password']
            if not bcrypt.checkpw(current_password.encode(), stored_password.encode()):
                st.error("Current password is incorrect")
                return
                
            # Update password
            hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
            config['credentials']['usernames'][st.session_state.username]['password'] = hashed_password
            
            # Save updated config
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file)
                
            st.success("Password updated successfully")

def create_and_connect_to_sample_db():
    """Create a sample database and connect to it"""
    try:
        # Create and populate database
        db_path = create_and_populate_sample_db()
        
        # Connect to the new database
        engine = create_engine(f'sqlite:///{db_path}')
        schema = get_db_schema(engine)
        
        # Update session state
        st.session_state.db_engine = engine
        st.session_state.schema = schema
        st.session_state.db_initialized = True
        st.session_state.current_temp_file = db_path
        
        # Add to temp files list for cleanup
        temp_files.append(db_path)
        
        st.success("Successfully created and connected to sample cafe database!")
        st.markdown("""
        ### Sample Database Created
        This database contains information about a coffee shop with:
        - Employee data (names, positions, salaries, etc.)
        - Product information (coffees, teas, food items)
        - Sales records connecting employees and products
        
        Try asking questions like:
        - "Show me a pie chart of salary distribution by position"
        - "Give me a bar chart of product sales by category"
        - "Display the correlation between performance rating and salary"
        """)
        
    except Exception as e:
        st.error(f"Error creating sample database: {str(e)}")
        st.session_state.db_initialized = False

def create_and_populate_sample_db():
    """Create a new sample database and populate it with realistic data"""
    # Create a temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    conn = sqlite3.connect(temp_db.name)
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute('''
    CREATE TABLE employees (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      position TEXT NOT NULL,
      salary REAL,
      hire_date TEXT,
      shift_preference TEXT,
      barista_certification BOOLEAN,
      years_experience REAL,
      performance_rating REAL,
      feedback TEXT,
      contact_number TEXT,
      email TEXT,
      emergency_contact TEXT,
      is_full_time BOOLEAN
    )
    ''')
    
    # Create products table
    cursor.execute('''
    CREATE TABLE products (
      product_id INTEGER PRIMARY KEY AUTOINCREMENT,
      product_name TEXT NOT NULL,
      category TEXT NOT NULL,
      price REAL NOT NULL,
      cost REAL NOT NULL,
      stock_quantity INTEGER NOT NULL,
      supplier TEXT,
      reorder_level INTEGER
    )
    ''')
    
    # Create sales table that relates to both employees and products
    cursor.execute('''
    CREATE TABLE sales (
      sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
      employee_id INTEGER,
      product_id INTEGER,
      quantity INTEGER NOT NULL,
      sale_date TEXT NOT NULL,
      total_amount REAL NOT NULL,
      payment_method TEXT,
      FOREIGN KEY (employee_id) REFERENCES employees (id),
      FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Sample data for employees
    employees = [
        ("Emma Thompson", "Senior Barista", 36262.94, "2019-05-12", "Morning", 1, 4.2, 4.8, "Excellent team player, great with customers", "555-123-4567", "emma@coffeeworld.com", "John Thompson: 555-987-6543", 1),
        ("Liam Chen", "Store Manager", 59935.51, "2018-02-28", "Morning", 1, 6.5, 4.9, "Outstanding leadership skills", "555-234-5678", "liam@coffeeworld.com", "Lin Chen: 555-876-5432", 1),
        ("Ava Patel", "Shift Supervisor", 40777.79, "2020-01-15", "Evening", 1, 3.7, 4.5, "Reliable and efficient", "555-345-6789", "ava@coffeeworld.com", "Raj Patel: 555-765-4321", 1),
        ("Noah Rodriguez", "Barista", 28990.20, "2021-07-20", "Afternoon", 1, 1.5, 3.9, "Quick learner, needs more customer service training", "555-456-7890", "noah@coffeeworld.com", "Maria Rodriguez: 555-654-3210", 1),
        ("Sofia Rodriguez", "Senior Barista", 34467.67, "2019-11-05", "Afternoon", 1, 3.8, 4.6, "Excellent latte artist", "555-567-8901", "sofia@coffeeworld.com", "Carlos Rodriguez: 555-543-2109", 1),
        ("Jackson Kim", "Barista", 27658.30, "2022-03-10", "Evening", 0, 0.8, 3.5, "Improving steadily", "555-678-9012", "jackson@coffeeworld.com", "Min Kim: 555-432-1098", 0),
        ("Olivia Johnson", "Barista", 28120.15, "2021-10-18", "Morning", 1, 1.2, 3.7, "Great with customers", "555-789-0123", "olivia@coffeeworld.com", "David Johnson: 555-321-0987", 1),
        ("Lucas Martinez", "Kitchen Staff", 26345.92, "2022-01-05", "Morning", 0, 1.0, 3.8, "Hard worker, efficient", "555-890-1234", "lucas@coffeeworld.com", "Ana Martinez: 555-210-9876", 0),
        ("Amelia Wilson", "Cashier", 25780.45, "2022-04-20", "Afternoon", 0, 0.6, 3.4, "Quick and accurate", "555-901-2345", "amelia@coffeeworld.com", "Robert Wilson: 555-109-8765", 0),
        ("Mason Williams", "Kitchen Staff", 29080.83, "2021-05-15", "Evening", 0, 2.2, 4.1, "Creative with food presentation", "555-012-3456", "mason@coffeeworld.com", "Sarah Williams: 555-098-7654", 1),
        ("Harper Garcia", "Shift Supervisor", 39120.50, "2020-06-10", "Evening", 1, 3.0, 4.3, "Great at resolving conflicts", "555-123-5678", "harper@coffeeworld.com", "Miguel Garcia: 555-987-6543", 1),
        ("Elijah Brown", "Barista", 27400.00, "2022-02-15", "Afternoon", 0, 0.9, 3.6, "Friendly and punctual", "555-234-5679", "elijah@coffeeworld.com", "Jessica Brown: 555-876-5433", 0),
        ("Abigail Taylor", "Cashier", 26100.75, "2022-03-01", "Morning", 0, 0.7, 3.5, "Detail-oriented", "555-345-6780", "abigail@coffeeworld.com", "Michael Taylor: 555-765-4322", 0),
        ("Benjamin Smith", "Senior Barista", 33750.25, "2020-04-10", "Morning", 1, 2.9, 4.4, "Coffee expert, great trainer", "555-456-7891", "benjamin@coffeeworld.com", "Emily Smith: 555-654-3211", 1),
        ("Isabella Davis", "Kitchen Staff", 27980.60, "2021-08-15", "Afternoon", 0, 1.6, 3.9, "Fast and consistent", "555-567-8902", "isabella@coffeeworld.com", "Anthony Davis: 555-543-2100", 1)
    ]
    
    cursor.executemany('''
    INSERT INTO employees (name, position, salary, hire_date, shift_preference, barista_certification, 
                          years_experience, performance_rating, feedback, contact_number, 
                          email, emergency_contact, is_full_time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', employees)
    
    # Sample data for products
    products = [
        ("Espresso", "Hot Coffee", 3.50, 0.75, 100, "Premium Coffee Suppliers", 20),
        ("Cappuccino", "Hot Coffee", 4.50, 1.00, 100, "Premium Coffee Suppliers", 20),
        ("Latte", "Hot Coffee", 4.75, 1.00, 100, "Premium Coffee Suppliers", 20),
        ("Americano", "Hot Coffee", 3.75, 0.80, 100, "Premium Coffee Suppliers", 20),
        ("Mocha", "Hot Coffee", 5.00, 1.25, 100, "Premium Coffee Suppliers", 20),
        ("Cold Brew", "Cold Coffee", 4.25, 1.00, 75, "Premium Coffee Suppliers", 15),
        ("Iced Latte", "Cold Coffee", 5.00, 1.10, 75, "Premium Coffee Suppliers", 15),
        ("Frappuccino", "Cold Coffee", 5.50, 1.50, 50, "Premium Coffee Suppliers", 10),
        ("Green Tea", "Tea", 3.25, 0.50, 80, "TeaLeaf Co.", 15),
        ("Black Tea", "Tea", 3.25, 0.50, 80, "TeaLeaf Co.", 15),
        ("Chai Latte", "Tea", 4.50, 1.00, 60, "TeaLeaf Co.", 12),
        ("Chocolate Chip Cookie", "Bakery", 2.50, 0.60, 40, "Local Bakery", 10),
        ("Blueberry Muffin", "Bakery", 3.25, 0.75, 35, "Local Bakery", 8),
        ("Croissant", "Bakery", 3.00, 0.70, 30, "Local Bakery", 8),
        ("Chicken Sandwich", "Food", 6.50, 2.50, 25, "Fresh Foods Inc.", 5),
        ("Avocado Toast", "Food", 7.00, 2.75, 20, "Fresh Foods Inc.", 5),
        ("Fruit Cup", "Food", 4.50, 1.50, 15, "Fresh Foods Inc.", 3),
        ("Bottled Water", "Cold Drinks", 2.00, 0.50, 100, "Beverage Distributors", 20),
        ("Iced Tea", "Cold Drinks", 3.00, 0.60, 80, "Beverage Distributors", 15),
        ("Lemonade", "Cold Drinks", 3.50, 0.75, 60, "Beverage Distributors", 12)
    ]
    
    cursor.executemany('''
    INSERT INTO products (product_name, category, price, cost, stock_quantity, supplier, reorder_level)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', products)
    
    # Generate sample sales data
    sales = []
    # Get the IDs after insert
    cursor.execute("SELECT id FROM employees")
    employee_ids = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT product_id FROM products")
    product_ids = [row[0] for row in cursor.fetchall()]
    
    # Generate sales for the last 30 days
    payment_methods = ["Credit Card", "Cash", "Mobile Payment", "Gift Card"]
    
    for day in range(30, 0, -1):
        sale_date = (datetime.datetime.now() - datetime.timedelta(days=day)).strftime("%Y-%m-%d")
        
        # Generate between 30-50 sales per day
        for _ in range(random.randint(30, 50)):
            employee_id = random.choice(employee_ids)
            product_id = random.choice(product_ids)
            
            # Get the product price
            cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
            price = cursor.fetchone()[0]
            
            quantity = random.randint(1, 3)
            total = price * quantity
            payment_method = random.choice(payment_methods)
            
            sales.append((employee_id, product_id, quantity, sale_date, total, payment_method))
    
    cursor.executemany('''
    INSERT INTO sales (employee_id, product_id, quantity, sale_date, total_amount, payment_method)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', sales)
    
    # Commit and close
    conn.commit()
    conn.close()
    
    return temp_db.name

# Run the application when the script is executed directly
if __name__ == "__main__":
    main()

