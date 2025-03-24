import sqlite3
from pathlib import Path
import pandas as pd
from .schema import TableSchema

class DatabaseManager:
    """Manager class for database operations with immutable data policy"""
    
    def __init__(self, db_path: Path):
        """Initialize database manager
        
        Args:
            db_path (Path): Path to the SQLite database file
        """
        self.db_path = db_path

    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)

    def _close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database
        
        Args:
            table_name (str): Name of the table to check
            
        Returns:
            bool: True if table exists, False otherwise
        """
        try:
            self._connect()
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            return cursor.fetchone() is not None
        except Exception as e:
            raise Exception(f"Error checking table existence: {str(e)}")
        finally:
            self._close()

    def setup_table(self, schema: TableSchema):
        """Initialize database by creating table and indexes"""
        try:
            self._connect()
            self.schema = schema
                
            # Create table and indexes by keys if it doesn't exist
            self.conn.execute(self.schema.to_sql())
            for index_sql in self.schema.to_index_sql():
                self.conn.execute(index_sql)
            self.conn.commit()

        except Exception as e:
            raise Exception(f"Database setup error: {str(e)}")
        finally:
            self._close()

    def delete_table(self, table_name: str):
        """Delete a table from the database
        
        Args:
            table_name (str): Name of the table to delete
        """
        try:
            self._connect()
            self.conn.execute(f"DROP TABLE {table_name}")
            self.conn.commit()
        except Exception as e:
            raise Exception(f"Error deleting table: {str(e)}")
        finally:
            self._close()

    def insert(self, data: pd.DataFrame, table_name: str):
        """Insert data into the table
        
        Args:
            data (pd.DataFrame): DataFrame containing the data to insert
        """
        try:
            self._connect()
            data.to_sql(table_name, self.conn, if_exists='append', index=False)
            self.conn.commit()
        except Exception as e:
            raise Exception(f"Error inserting data: {str(e)}")
        finally:
            self._close()

    def query(self, sql_query: str) -> pd.DataFrame:
        """Query data from the table
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            pd.DataFrame: Query results
        """
        try:
            self._connect()
            return pd.read_sql_query(sql_query, self.conn)
            pass
        except Exception as e:
            raise Exception(f"Error querying data: {str(e)}")
        finally:
            self._close() 

