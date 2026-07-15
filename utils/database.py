import sqlite3
import re
import pandas as pd
from typing import Tuple, Dict, Any, Optional

def sanitize_column_name(col: str) -> str:
    """
    Sanitize column names to be valid SQLite identifiers.
    Converts to lowercase, replaces spaces and special characters with underscores,
    handles currency and percentage symbols, and removes consecutive underscores.
    """
    col = str(col).strip().lower()
    # Replace currency symbols and percentage
    col = col.replace("$", "usd").replace("€", "eur").replace("£", "gbp").replace("%", "percent")
    # Replace any non-alphanumeric char with underscore
    col = re.sub(r'[^a-z0-9_]', '_', col)
    # Replace multiple consecutive underscores
    col = re.sub(r'_+', '_', col)
    # Strip leading/trailing underscores
    col = col.strip('_')
    # SQL identifiers cannot start with numbers or be empty
    if not col:
        col = "col"
    elif col[0].isdigit():
        col = f"col_{col}"
    return col

def sanitize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Sanitize all column names in the dataframe.
    Returns the new dataframe and a mapping from original to sanitized column names.
    """
    df_clean = df.copy()
    mapping = {}
    seen_cols = set()
    new_cols = []
    
    for col in df.columns:
        sanitized = sanitize_column_name(col)
        original_sanitized = sanitized
        counter = 2
        while sanitized in seen_cols:
            sanitized = f"{original_sanitized}_{counter}"
            counter += 1
        seen_cols.add(sanitized)
        mapping[col] = sanitized
        new_cols.append(sanitized)
        
    df_clean.columns = new_cols
    return df_clean, mapping

class DatabaseManager:
    """
    Manages loading and querying tabular data using an in-memory SQLite database.
    """
    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.table_name = "dataset"
        self.original_columns_map: Dict[str, str] = {} # original -> sanitized
        self.sanitized_columns_map: Dict[str, str] = {} # sanitized -> original
        self.has_data = False
        self.raw_row_count = 0

    def load_file(self, file_path_or_buffer: Any, file_type: str) -> Tuple[bool, Optional[str]]:
        """
        Loads a CSV or Excel file into the SQLite database.
        Returns (success, error_message).
        """
        try:
            if file_type == "csv":
                df = pd.read_csv(file_path_or_buffer)
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(file_path_or_buffer)
            else:
                return False, f"Unsupported file type: {file_type}"
            
            if df.empty:
                return False, "The uploaded file is empty."
                
            self.raw_row_count = len(df)
            
            # Clean and sanitize column names for SQL compliance
            df_clean, mapping = sanitize_dataframe(df)
            self.original_columns_map = mapping
            self.sanitized_columns_map = {v: k for k, v in mapping.items()}
            
            # Save to SQLite
            df_clean.to_sql(self.table_name, self.conn, if_exists="replace", index=False)
            self.has_data = True
            
            return True, None
        except Exception as e:
            return False, f"Failed to load file: {str(e)}"

    def execute_query(self, sql_query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Executes a SQL query on the database.
        Returns a tuple of (DataFrame result, error message if any).
        """
        if not self.has_data:
            return None, "No dataset loaded in database."
        try:
            # Basic validation to prevent write operations (read-only SQLite check)
            query_stripped = sql_query.strip().lower()
            forbidden_keywords = ["drop", "delete", "insert", "update", "alter", "create", "replace", "truncate"]
            for keyword in forbidden_keywords:
                # Use regex word boundaries to avoid false positives (e.g. column named 'create_date')
                if re.search(r'\b' + keyword + r'\b', query_stripped):
                    return None, f"Query rejected: Write operations are not allowed (found forbidden keyword: '{keyword}')"
            
            df_res = pd.read_sql_query(sql_query, self.conn)
            return df_res, None
        except Exception as e:
            return None, str(e)

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Returns schema information about the loaded table,
        including data types and a few sample rows to guide the LLM.
        """
        if not self.has_data:
            return {}
            
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        columns_info = cursor.fetchall() # list of tuples: (cid, name, type, notnull, dflt_value, pk)
        
        # Get sample rows
        cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 3")
        sample_rows = cursor.fetchall()
        
        columns = []
        for col in columns_info:
            col_name = col[1]
            columns.append({
                "name": col_name,
                "type": col[2],
                "original_name": self.sanitized_columns_map.get(col_name, col_name)
            })
            
        return {
            "table_name": self.table_name,
            "columns": columns,
            "sample_rows": sample_rows,
            "row_count": self.raw_row_count
        }

    def get_schema_summary_text(self) -> str:
        """
        Generates a text summary of the schema suitable for an LLM prompt.
        """
        info = self.get_schema_info()
        if not info:
            return "No schema available."
            
        summary = f"Table Name: {info['table_name']}\n"
        summary += f"Total Rows: {info['row_count']}\n"
        summary += "Columns:\n"
        for col in info['columns']:
            summary += f" - {col['name']} ({col['type']}) [Original name: '{col['original_name']}']\n"
            
        summary += "\nSample Data (First 3 rows):\n"
        if info['sample_rows']:
            # Create a nice markdown table representation for the LLM
            df_sample = pd.read_sql_query(f"SELECT * FROM {self.table_name} LIMIT 3", self.conn)
            summary += df_sample.to_markdown(index=False)
        else:
            summary += "No rows available."
            
        return summary
