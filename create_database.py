import sqlite3
import os

# 1. Ensure the directory exists so it doesn't crash
os.makedirs("database", exist_ok=True)

# 2. Connect to (or create) the database
conn = sqlite3.connect("database/patients.db")
cursor = conn.cursor()

# 3. Create Staff table
cursor.execute("""
CREATE TABLE IF NOT EXISTS staff(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT
)
""")

# 4. Create Patient predictions table
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age TEXT,
    gender TEXT,
    race TEXT,
    time_in_hospital INTEGER,
    prediction INTEGER,
    probability REAL
)
""")

# 5. Insert Default doctor login (Check if exists first to avoid duplicates if run twice)
cursor.execute("SELECT * FROM staff WHERE username='doctor'")
if not cursor.fetchone():
    cursor.execute("INSERT INTO staff(username,password) VALUES('doctor','1234')")

conn.commit()
conn.close()

print("✅ Database created successfully at database/patients.db!")
