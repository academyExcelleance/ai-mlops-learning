import psycopg2

conn = psycopg2.connect(
    host="airflow-db.c0lk2aw6qbxi.us-east-1.rds.amazonaws.com",
    database="airflow-db",
    user="postgres",
    password="postgres",
    port=5432
)

cur = conn.cursor()
cur.execute("SELECT datname FROM pg_database;")
databases = cur.fetchall()
print(databases)

cur.close()
conn.close()
