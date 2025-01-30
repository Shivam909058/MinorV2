import openai
import os
import tempfile
from typing import Optional, Tuple, Dict

# Set your OpenAI API key
OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

temp_files = [] 

def add_temp_file(file_path: str):
    temp_files.append(file_path)

def cleanup_temp_files():
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def clean_sql_query(query: str) -> str:
    """Clean SQL query by removing markdown formatting and unnecessary whitespace."""
    query = query.replace('```sql', '').replace('```', '')
    query = '\n'.join(line.strip() for line in query.splitlines() if line.strip())
    return query

def describe_database(schema: Dict) -> str:
    """Describe the database schema in natural language."""
    tables_info = "\n".join([f"Table '{table}' with columns: {', '.join(columns)}" 
                            for table, columns in schema.items()])
    prompt = f"The database has the following structure:\n\n{tables_info}\n\n" \
            "Provide a natural language summary of this database structure, " \
            "explaining what kind of data it might contain and how the tables might be related."
    return generate_response(prompt)

def generate_sql_query(prompt: str, schema: dict) -> Tuple[Optional[str], str]:
    """Generate SQL query from natural language prompt."""
    try:
        # First, check if this is a general question about the database
        if any(keyword in prompt.lower() 
               for keyword in ['what is', 'tell me about', 'describe', 'explain', 'show me']):
            return None, describe_database(schema)

        system_message = """You are a SQL expert. When given a question about a database:
        1. If the question requires data retrieval, provide a SQL query
        2. If it's a general question about the database structure, provide a natural explanation
        3. Format SQL responses as:
           QUERY: <the SQL query>
           EXPLANATION: <brief explanation of what the query does>
        4. Keep queries simple and efficient
        5. Use proper SQL syntax and conventions
        6. Only use tables and columns that exist in the schema"""
        
        schema_context = f"Database schema: {schema}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "system", "content": schema_context},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response['choices'][0]['message']['content']
        
        # Extract query and explanation
        if "QUERY:" in response_text:
            parts = response_text.split("QUERY:", 1)[1].split("EXPLANATION:")
            query = clean_sql_query(parts[0].strip())
            explanation = parts[1].strip() if len(parts) > 1 else ""
            return query, explanation
        
        # If no query is needed, return the response as explanation
        return None, response_text

    except Exception as e:
        return None, f"Error generating response: {str(e)}"

def generate_response(prompt: str) -> str:
    """Generate a natural language response."""
    try:
        system_message = """You are a helpful database assistant. When explaining query results:
        1. Use clear and concise language
        2. Highlight key insights from the data
        3. Format numbers in a readable way
        4. Mention any interesting patterns or anomalies
        5. Keep explanations professional but conversational"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"