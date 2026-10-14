import sqlite3
import os

def setup_local_database() -> str:
    db_dir = "data"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "enterprise.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create employees table with COLLATE NOCASE to prevent case-sensitivity issues!
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT COLLATE NOCASE NOT NULL,
            salary REAL NOT NULL
        );
    """)
    
    # Seed mock data
    sample_employees = [
        ("Alice Smith", "Engineering", 95000.0),
        ("Bob Jones", "Marketing", 75000.0),
        ("Charlie Brown", "Engineering", 105000.0),
        ("Diana Prince", "Sales", 82000.0),
        ("Evan Wright", "Engineering", 110000.0)
    ]
    
    cursor.executemany("""
        INSERT INTO employees (name, department, salary)
        SELECT ?, ?, ? 
        WHERE NOT EXISTS (SELECT 1 FROM employees WHERE name = ?);
    """, [(name, dept, sal, name) for name, dept, sal in sample_employees])
    
    conn.commit()
    conn.close()
    
    print(f"[*] Local SQLite database ready at: {db_path}")
    return db_path

if __name__ == "__main__":
    setup_local_database()