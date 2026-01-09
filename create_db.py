import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Database connection parameters
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def create_crm_ai_db_table():
    """Create the crm-ai-db table with sid as primary key and Completed column."""
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        cursor = conn.cursor()
        
        # Create table with sid as primary key and Completed with default value False
        create_table_query = """
        CREATE TABLE IF NOT EXISTS "crm-ai-db" (
            sid VARCHAR(255) PRIMARY KEY,
            "Completed" BOOLEAN DEFAULT FALSE
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        
        print("✅ Table 'crm-ai-db' created successfully!")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def create_call_contexts_table():
    """Create the call_contexts table to store call context data across workers."""
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        cursor = conn.cursor()
        
        # Create call_contexts table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS call_contexts (
            call_sid VARCHAR(255) PRIMARY KEY,
            phone_number VARCHAR(20) NOT NULL,
            app_name VARCHAR(255),
            reason TEXT,
            language VARCHAR(50),
            client_name VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        );
        """
        
        cursor.execute(create_table_query)
        
        # Create index on call_sid for faster lookups
        create_index_query = """
        CREATE INDEX IF NOT EXISTS idx_call_contexts_call_sid ON call_contexts(call_sid);
        """
        cursor.execute(create_index_query)
        
        conn.commit()
        
        print("✅ Table 'call_contexts' created successfully!")
        
        cursor.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_crm_ai_db_table()
    create_call_contexts_table()
