import sqlite3
from datetime import datetime, timedelta
import random
def create_and_populate_sample_db():
    """Create a new sample database and populate it with realistic data"""
    import sqlite3
    import datetime
    import random
    import tempfile
    
    # Create a temporary database file
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    conn = sqlite3.connect(temp_db.name)
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute('''
    CREATE TABLE employees (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      position TEXT NOT NULL,
      salary REAL,
      hire_date TEXT,
      shift_preference TEXT,
      barista_certification BOOLEAN,
      years_experience REAL,
      performance_rating REAL,
      feedback TEXT,
      contact_number TEXT,
      email TEXT,
      emergency_contact TEXT,
      is_full_time BOOLEAN
    )
    ''')
    
    # Create products table
    cursor.execute('''
    CREATE TABLE products (
      product_id INTEGER PRIMARY KEY AUTOINCREMENT,
      product_name TEXT NOT NULL,
      category TEXT NOT NULL,
      price REAL NOT NULL,
      cost REAL NOT NULL,
      stock_quantity INTEGER NOT NULL,
      supplier TEXT,
      reorder_level INTEGER
    )
    ''')
    
    # Create sales table that relates to both employees and products
    cursor.execute('''
    CREATE TABLE sales (
      sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
      employee_id INTEGER,
      product_id INTEGER,
      quantity INTEGER NOT NULL,
      sale_date TEXT NOT NULL,
      total_amount REAL NOT NULL,
      payment_method TEXT,
      FOREIGN KEY (employee_id) REFERENCES employees (id),
      FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Sample data for employees
    employees = [
        ("Emma Thompson", "Senior Barista", 36262.94, "2019-05-12", "Morning", 1, 4.2, 4.8, "Excellent team player, great with customers", "555-123-4567", "emma@coffeeworld.com", "John Thompson: 555-987-6543", 1),
        ("Liam Chen", "Store Manager", 59935.51, "2018-02-28", "Morning", 1, 6.5, 4.9, "Outstanding leadership skills", "555-234-5678", "liam@coffeeworld.com", "Lin Chen: 555-876-5432", 1),
        ("Ava Patel", "Shift Supervisor", 40777.79, "2020-01-15", "Evening", 1, 3.7, 4.5, "Reliable and efficient", "555-345-6789", "ava@coffeeworld.com", "Raj Patel: 555-765-4321", 1),
        ("Noah Rodriguez", "Barista", 28990.20, "2021-07-20", "Afternoon", 1, 1.5, 3.9, "Quick learner, needs more customer service training", "555-456-7890", "noah@coffeeworld.com", "Maria Rodriguez: 555-654-3210", 1),
        ("Sofia Rodriguez", "Senior Barista", 34467.67, "2019-11-05", "Afternoon", 1, 3.8, 4.6, "Excellent latte artist", "555-567-8901", "sofia@coffeeworld.com", "Carlos Rodriguez: 555-543-2109", 1),
        ("Jackson Kim", "Barista", 27658.30, "2022-03-10", "Evening", 0, 0.8, 3.5, "Improving steadily", "555-678-9012", "jackson@coffeeworld.com", "Min Kim: 555-432-1098", 0),
        ("Olivia Johnson", "Barista", 28120.15, "2021-10-18", "Morning", 1, 1.2, 3.7, "Great with customers", "555-789-0123", "olivia@coffeeworld.com", "David Johnson: 555-321-0987", 1),
        ("Lucas Martinez", "Kitchen Staff", 26345.92, "2022-01-05", "Morning", 0, 1.0, 3.8, "Hard worker, efficient", "555-890-1234", "lucas@coffeeworld.com", "Ana Martinez: 555-210-9876", 0),
        ("Amelia Wilson", "Cashier", 25780.45, "2022-04-20", "Afternoon", 0, 0.6, 3.4, "Quick and accurate", "555-901-2345", "amelia@coffeeworld.com", "Robert Wilson: 555-109-8765", 0),
        ("Mason Williams", "Kitchen Staff", 29080.83, "2021-05-15", "Evening", 0, 2.2, 4.1, "Creative with food presentation", "555-012-3456", "mason@coffeeworld.com", "Sarah Williams: 555-098-7654", 1),
        ("Harper Garcia", "Shift Supervisor", 39120.50, "2020-06-10", "Evening", 1, 3.0, 4.3, "Great at resolving conflicts", "555-123-5678", "harper@coffeeworld.com", "Miguel Garcia: 555-987-6543", 1),
        ("Elijah Brown", "Barista", 27400.00, "2022-02-15", "Afternoon", 0, 0.9, 3.6, "Friendly and punctual", "555-234-5679", "elijah@coffeeworld.com", "Jessica Brown: 555-876-5433", 0),
        ("Abigail Taylor", "Cashier", 26100.75, "2022-03-01", "Morning", 0, 0.7, 3.5, "Detail-oriented", "555-345-6780", "abigail@coffeeworld.com", "Michael Taylor: 555-765-4322", 0),
        ("Benjamin Smith", "Senior Barista", 33750.25, "2020-04-10", "Morning", 1, 2.9, 4.4, "Coffee expert, great trainer", "555-456-7891", "benjamin@coffeeworld.com", "Emily Smith: 555-654-3211", 1),
        ("Isabella Davis", "Kitchen Staff", 27980.60, "2021-08-15", "Afternoon", 0, 1.6, 3.9, "Fast and consistent", "555-567-8902", "isabella@coffeeworld.com", "Anthony Davis: 555-543-2100", 1)
    ]
    
    cursor.executemany('''
    INSERT INTO employees (name, position, salary, hire_date, shift_preference, barista_certification, 
                          years_experience, performance_rating, feedback, contact_number, 
                          email, emergency_contact, is_full_time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', employees)
    
    # Sample data for products
    products = [
        ("Espresso", "Hot Coffee", 3.50, 0.75, 100, "Premium Coffee Suppliers", 20),
        ("Cappuccino", "Hot Coffee", 4.50, 1.00, 100, "Premium Coffee Suppliers", 20),
        ("Latte", "Hot Coffee", 4.75, 1.00, 100, "Premium Coffee Suppliers", 20),
        ("Americano", "Hot Coffee", 3.75, 0.80, 100, "Premium Coffee Suppliers", 20),
        ("Mocha", "Hot Coffee", 5.00, 1.25, 100, "Premium Coffee Suppliers", 20),
        ("Cold Brew", "Cold Coffee", 4.25, 1.00, 75, "Premium Coffee Suppliers", 15),
        ("Iced Latte", "Cold Coffee", 5.00, 1.10, 75, "Premium Coffee Suppliers", 15),
        ("Frappuccino", "Cold Coffee", 5.50, 1.50, 50, "Premium Coffee Suppliers", 10),
        ("Green Tea", "Tea", 3.25, 0.50, 80, "TeaLeaf Co.", 15),
        ("Black Tea", "Tea", 3.25, 0.50, 80, "TeaLeaf Co.", 15),
        ("Chai Latte", "Tea", 4.50, 1.00, 60, "TeaLeaf Co.", 12),
        ("Chocolate Chip Cookie", "Bakery", 2.50, 0.60, 40, "Local Bakery", 10),
        ("Blueberry Muffin", "Bakery", 3.25, 0.75, 35, "Local Bakery", 8),
        ("Croissant", "Bakery", 3.00, 0.70, 30, "Local Bakery", 8),
        ("Chicken Sandwich", "Food", 6.50, 2.50, 25, "Fresh Foods Inc.", 5),
        ("Avocado Toast", "Food", 7.00, 2.75, 20, "Fresh Foods Inc.", 5),
        ("Fruit Cup", "Food", 4.50, 1.50, 15, "Fresh Foods Inc.", 3),
        ("Bottled Water", "Cold Drinks", 2.00, 0.50, 100, "Beverage Distributors", 20),
        ("Iced Tea", "Cold Drinks", 3.00, 0.60, 80, "Beverage Distributors", 15),
        ("Lemonade", "Cold Drinks", 3.50, 0.75, 60, "Beverage Distributors", 12)
    ]
    
    cursor.executemany('''
    INSERT INTO products (product_name, category, price, cost, stock_quantity, supplier, reorder_level)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', products)
    
    # Generate sample sales data
    sales = []
    # Get the IDs after insert
    cursor.execute("SELECT id FROM employees")
    employee_ids = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT product_id FROM products")
    product_ids = [row[0] for row in cursor.fetchall()]
    
    # Generate sales for the last 30 days
    payment_methods = ["Credit Card", "Cash", "Mobile Payment", "Gift Card"]
    
    for day in range(30, 0, -1):
        sale_date = (datetime.datetime.now() - datetime.timedelta(days=day)).strftime("%Y-%m-%d")
        
        # Generate between 30-50 sales per day
        for _ in range(random.randint(30, 50)):
            employee_id = random.choice(employee_ids)
            product_id = random.choice(product_ids)
            
            # Get the product price
            cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
            price = cursor.fetchone()[0]
            
            quantity = random.randint(1, 3)
            total = price * quantity
            payment_method = random.choice(payment_methods)
            
            sales.append((employee_id, product_id, quantity, sale_date, total, payment_method))
    
    cursor.executemany('''
    INSERT INTO sales (employee_id, product_id, quantity, sale_date, total_amount, payment_method)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', sales)
    
    # Commit and close
    conn.commit()
    conn.close()
    
    return temp_db.name

create_and_populate_sample_db()

if __name__ == "__main__":
    create_and_populate_sample_db()
    print("Coffee shop database created successfully!")