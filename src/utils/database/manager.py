import sqlite3
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict, Any
from .schema import TableSchema

class DatabaseManager:
    """Manager class for database operations with immutable data policy"""
    
    def __init__(self, db_path: Path, schema: TableSchema):
        """Initialize database manager
        
        Args:
            db_path (Path): Path to the SQLite database file
            schema (TableSchema): The schema to use for this database manager
        """
        self.db_path = db_path
        self.schema = schema
        self.conn: Optional[sqlite3.Connection] = None

    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)

    def _close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def close(self):
        """Public method to close the database connection"""
        self._close()

    def setup(self):
        """Initialize database by creating table and indexes"""
        try:
            self._connect()
            self.conn.execute(self.schema.to_sql())
            for index_sql in self.schema.to_index_sql():
                self.conn.execute(index_sql)
            self.conn.commit()
        except Exception as e:
            raise Exception(f"Database setup error: {str(e)}")
        finally:
            self._close()

    def insert(self, data: pd.DataFrame):
        """Insert data into the table
        
        Args:
            data (pd.DataFrame): DataFrame containing the data to insert
        """
        try:
            self._connect()
            data.to_sql(self.schema.name, self.conn, if_exists='append', index=False)
            self.conn.commit()
        except Exception as e:
            raise Exception(f"Error inserting data: {str(e)}")
        finally:
            self._close()

    def query(self, 
             columns: List[str] = None,
             where_conditions: Dict[str, Any] = None) -> pd.DataFrame:
        """Query data from the table
        
        Args:
            columns (List[str], optional): List of columns to select
            where_conditions (Dict[str, Any], optional): WHERE clause conditions
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            self._connect()
            # Implementation for query operation
            # This is a placeholder - actual implementation would depend on specific needs
            pass
        except Exception as e:
            raise Exception(f"Error querying data: {str(e)}")
        finally:
            self._close() 