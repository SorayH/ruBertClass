import sqlite3

def res_db(tb_name):

    conn = sqlite3.connect("neur_type.db")
    cursor = conn.cursor()

    res = []

    cursor.execute("SELECT * FROM " + tb_name)
    rows = cursor.fetchall()

    for row in rows:
        res.append(row[0])

    return res