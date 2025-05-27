import mysql.connector
from mysql.connector import Error
from datetime import datetime, date

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="qiwoqqwu",
            database="miniproject"
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def convert_datetime_to_str(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (int, float)):
        return str(obj)
    return obj

def execute_query(query, params=None):
    connection = get_db_connection()
    if connection is None:
        return None
    
    try:
        cursor = connection.cursor(dictionary=True)
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if query.strip().upper().startswith('SELECT'):
            result = cursor.fetchall()
            # Convert datetime and date fields to string
            for row in result:
                for key, value in row.items():
                    row[key] = convert_datetime_to_str(value)
        else:
            connection.commit()
            result = cursor.rowcount
        
        return result
    except Error as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close() 