import sqlite3
from datetime import datetime, timedelta
import random

def create_sample_database():
    # Connect to database
    conn = sqlite3.connect('coffee_shop.db')
    cursor = conn.cursor()

    # Create employees table with comprehensive details
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
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
    );
    ''')

    # Sample data
    positions = ['Barista', 'Senior Barista', 'Shift Supervisor', 'Assistant Manager', 'Store Manager', 
                'Pastry Chef', 'Cashier', 'Kitchen Staff', 'Cleaner']
    
    shifts = ['Morning', 'Afternoon', 'Evening', 'Flexible']
    
    feedback_templates = [
        "Excellent customer service skills. {}", 
        "Shows great initiative in {}.",
        "Needs improvement in {}. Otherwise performing well.",
        "Consistently {} in their role.",
        "Outstanding performance in {}.",
        "Could benefit from additional training in {}.",
        "Demonstrates exceptional skills in {}.",
        "Very reliable and {}.",
        "Great team player, especially when {}."
    ]

    feedback_details = [
        "handling rush hours",
        "maintaining cleanliness",
        "coffee preparation",
        "customer interactions",
        "team collaboration",
        "following procedures",
        "training new staff",
        "managing inventory",
        "problem-solving",
        "attention to detail"
    ]

    # Generate 30 employees with realistic data
    employees = []
    start_date = datetime(2018, 1, 1)
    end_date = datetime.now()

    names = [
        "Emma Thompson", "Liam Chen", "Sofia Rodriguez", "Mason Williams", "Ava Patel",
        "Noah Kim", "Isabella Martinez", "Lucas Johnson", "Olivia Lee", "Ethan Brown",
        "Mia Garcia", "Alexander Wright", "Charlotte Davis", "William Taylor", "Sophia Anderson",
        "James Wilson", "Amelia Moore", "Benjamin White", "Harper Jackson", "Michael Lewis",
        "Evelyn Clark", "Daniel Martin", "Elizabeth Hall", "Joseph Young", "Victoria Adams",
        "David Miller", "Grace Turner", "Samuel Baker", "Chloe Parker", "Andrew Scott"
    ]

    for name in names:
        # Generate realistic employee data
        position = random.choice(positions)
        base_salary = {
            'Store Manager': (55000, 65000),
            'Assistant Manager': (45000, 52000),
            'Shift Supervisor': (35000, 42000),
            'Senior Barista': (32000, 38000),
            'Barista': (28000, 34000),
            'Pastry Chef': (35000, 45000),
            'Cashier': (26000, 32000),
            'Kitchen Staff': (27000, 33000),
            'Cleaner': (25000, 30000)
        }

        salary = round(random.uniform(*base_salary[position]), 2)
        hire_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        shift_pref = random.choice(shifts)
        is_barista_cert = True if position in ['Barista', 'Senior Barista', 'Shift Supervisor'] else False
        experience = round(random.uniform(0.5, 8.0), 1)
        rating = round(random.uniform(3.0, 5.0), 1)
        
        # Generate personalized feedback
        feedback = random.choice(feedback_templates).format(random.choice(feedback_details))
        
        # Generate contact information
        phone = f"(555) {random.randint(100,999)}-{random.randint(1000,9999)}"
        email = f"{name.lower().replace(' ', '.')}@coffeemail.com"
        emergency = f"(555) {random.randint(100,999)}-{random.randint(1000,9999)}"
        is_full_time = random.choice([True, True, True, False])  # 75% full-time

        employees.append((
            name, position, salary, hire_date.strftime('%Y-%m-%d'),
            shift_pref, is_barista_cert, experience, rating, feedback,
            phone, email, emergency, is_full_time
        ))

    # Insert all employees
    cursor.executemany('''
    INSERT INTO employees (
        name, position, salary, hire_date, shift_preference,
        barista_certification, years_experience, performance_rating,
        feedback, contact_number, email, emergency_contact, is_full_time
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    ''', employees)

    # Commit and close
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_sample_database()
    print("Coffee shop database created successfully!")