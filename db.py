# db.py

import mysql.connector

def connect():
    return mysql.connector.connect(
        host="ai_expense_tracker",
        user="root",
        password="root",
        database="expense_tracker"
    )

def insert_expense(description, amount, category):
    db = connect()
    cursor = db.cursor()
    query = "INSERT INTO expenses (description, amount, category) VALUES (%s, %s, %s)"
    values = (description, amount, category)
    cursor.execute(query, values)
    db.commit()
    cursor.close()
    db.close()

def get_all_expenses():
    db = connect()
    cursor = db.cursor()
    cursor.execute("SELECT description, amount, category, date FROM expenses ORDER BY date DESC")
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    return rows


