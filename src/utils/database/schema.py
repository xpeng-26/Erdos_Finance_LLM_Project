from dataclasses import dataclass
from typing import List

@dataclass
class Column:
    """Represents a database column definition"""
    name: str
    type: str
    description: str = ""

    def to_sql(self) -> str:
        """Convert column definition to SQL"""
        return f"{self.name} {self.type}"

@dataclass
class TableSchema:
    """Represents a database table schema"""
    name: str
    columns: List[Column]
    primary_keys: List[str]
    description: str = ""
    indexes: List[str] = None

    def __post_init__(self):
        if self.indexes is None:
            self.indexes = []

    def to_sql(self) -> str:
        """Convert table schema to SQL CREATE TABLE statement"""
        columns_sql = ', '.join(col.to_sql() for col in self.columns)
        pk_sql = f", PRIMARY KEY ({', '.join(self.primary_keys)})"
        return f"CREATE TABLE IF NOT EXISTS {self.name} ({columns_sql}{pk_sql})"

    def to_index_sql(self) -> List[str]:
        """Convert indexes to SQL CREATE INDEX statements"""
        return [f"CREATE INDEX IF NOT EXISTS idx_{self.name}_{idx} ON {self.name} ({idx})"
                for idx in self.indexes]

def create_schema(table_schema: str) -> TableSchema:
    """Create a new table schema
    
    Args:
        table_schema (str): type of the table
        
    Returns:
        TableSchema: The created schema
    """
    if table_schema == 'daily_prices':
        schema = TableSchema(
            name='daily_prices',
            description="Daily stock price data",
            columns=[
                Column('date', 'DATE', description="Trading date in UTC timezone"),
                Column('symbol', 'TEXT', description="Stock symbol"),
                Column('open', 'REAL', description="Opening price"),
                Column('high', 'REAL', description="Highest price"),
                Column('low', 'REAL', description="Lowest price"),
                Column('close', 'REAL', description="Closing price"),
                Column('volume', 'INTEGER', description="Trading volume"),
            ],
            primary_keys=['date', 'symbol'],
            indexes=['symbol', 'date']
        )
    elif table_schema == 'technical_factors':
        schema = TableSchema(
            name='technical_factors',
            description="Technical analysis factors",
            columns=[
                Column('date', 'DATE', description="Trading date"),
                Column('symbol', 'TEXT', description="Stock symbol"),
                Column('rsi_14', 'REAL', description="Relative Strength Index (14 periods)"),
                Column('roc_10', 'REAL', description="Rate of Change (10 periods)"),
                Column('mom_10', 'REAL', description="Momentum (10 periods)"),
            ],
            primary_keys=['date', 'symbol'],
            indexes=['symbol', 'date']
        )
    else:
        raise ValueError(f"Invalid table schema: {table_schema}")   
    return schema 