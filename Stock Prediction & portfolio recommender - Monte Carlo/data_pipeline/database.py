import os
import logging
from contextlib import contextmanager
from typing import List, Dict, Optional
import pandas as pd
import psycopg2
from psycopg2 import pool,extras
from psycopg2.extras import  execute_values
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self,
        host: str=None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        min_connections: int = 1,
        max_connections: int = 10):

        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.port = port or int(os.getenv('DB_PORT', '5432'))
        self.database = database or os.getenv('DB_NAME', 'stock_db')
        self.user = user or os.getenv('DB_USER', 'stock_user')
        self.password = password or os.getenv('DB_PASSWORD', 'stock_password')

        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                min_connections,
                max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"Connection pool created: {self.database}@{self.host}")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for connection ."""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """Context manager for cursor."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()