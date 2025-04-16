import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",         # 🔁 Replace with your credentials
        password="root",     # 🔁 Replace with your credentials
        database="expense_tracker"
    )

def insert_expense(description, amount, category):
    conn = get_connection()
    cursor = conn.cursor()
    query = "INSERT INTO expenses (description, amount, category) VALUES (%s, %s, %s)"
    cursor.execute(query, (description, amount, category))
    conn.commit()
    conn.close()

def fetch_all_expenses():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT description, amount, category, date FROM expenses ORDER BY date DESC")
    results = cursor.fetchall()
    conn.close()
    return results
