from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
import tempfile
import os
from typing import List, Dict, Optional
import pandas as pd
from .utils import add_temp_file

class DatabaseManager:
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.schema: Dict[str, List[str]] = {}

    def save_temp_file(self, file) -> str:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        add_temp_file(file_path)
        self.engine = create_engine(f"sqlite:///{file_path}")
        self.schema = self.get_schema(file_path)
        return file_path

    def connect_via_connection_string(self, connection_string: str):
        try:
            self.engine = create_engine(connection_string)
            self.schema = self.get_schema()
            return True, self.schema
        except Exception as e:
            return False, str(e)

    def get_schema(self, db_path: Optional[str] = None) -> Dict[str, List[str]]:
        if not self.engine:
            raise Exception("Database engine not initialized.")
        inspector = inspect(self.engine)
        schema = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema[table_name] = [col['name'] for col in columns]
        return schema

    def execute_query(self, query: str):
        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
        return {"columns": columns, "rows": rows}

    def format_result(self, result: Dict) -> str:
        try:
            df = pd.DataFrame(result['rows'], columns=result['columns'])
            if df.empty:
                return "No results found for this query."
            
            # Format numeric columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                df[col] = df[col].apply(lambda x: f"{x:,}" if pd.notnull(x) else '')
            
            # Add row numbers
            df.index = range(1, len(df) + 1)
            
            try:
                # Try using tabulate
                return df.to_markdown(index=True, tablefmt="pipe")
            except ImportError:
                # Fallback to basic string representation if tabulate is not available
                return df.to_string()
        except Exception as e:
            return f"Error formatting results: {str(e)}"