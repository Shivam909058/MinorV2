from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from .database import DatabaseManager
from .utils import generate_response, cleanup_temp_files, generate_sql_query
import os
import atexit
from pydantic import BaseModel

app = FastAPI(title="Database Assistant API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

atexit.register(cleanup_temp_files)
db_manager = DatabaseManager()

class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_db(query_request: QueryRequest):
    if not db_manager.engine:
        raise HTTPException(status_code=400, detail="No database connected.")
    try:
        # Generate SQL query from natural language
        sql_query, explanation = generate_sql_query(query_request.query, db_manager.schema)
        
        if not sql_query:
            return {
                "success": False,
                "explanation": explanation
            }
        
        try:
            # Execute the generated SQL query
            result = db_manager.execute_query(sql_query)
            formatted_result = db_manager.format_result(result)
            
            # Generate explanation of results
            result_explanation = generate_response(
                f"Explain these query results in natural language:\n{formatted_result}"
            )
            
            return {
                "success": True,
                "query": sql_query,
                "result": formatted_result,
                "explanation": result_explanation
            }
        except Exception as e:
            return {
                "success": False,
                "explanation": f"Error executing query: {str(e)}",
                "query": sql_query
            }
            
    except Exception as e:
        return {
            "success": False,
            "explanation": f"Error processing request: {str(e)}"
        }

@app.post("/upload-db/")
async def upload_db(file: UploadFile = File(...)):
    if file.content_type not in ["application/x-sqlite3", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    db_path = db_manager.save_temp_file(file)
    schema = db_manager.get_schema(db_path)
    description = generate_response(f"Describe the following database schema:\n{schema}")
    return {"message": "Database uploaded successfully.", "schema": schema, "description": description}

@app.post("/connect-db/")
async def connect_db(connection_string: str = Form(...)):
    success, schema_or_error = db_manager.connect_via_connection_string(connection_string)
    if not success:
        raise HTTPException(status_code=400, detail=schema_or_error)
    description = generate_response(f"Describe the following database schema:\n{schema_or_error}")
    return {"message": "Connected to database successfully.", "schema": schema_or_error, "description": description}