"""This file provides tips and suggestions for solving problems effectively."""

"""
Pythonic Thinking Workshop - Practice Exercises

Complete the following tasks using Pythonic approaches.
Try to apply the principles from the markdown file.
"""

# =============================================================================
# TASK 1: List Comprehensions
# =============================================================================

def task_1_non_pythonic():
    """Convert this to use list comprehension"""
    # Non-Pythonic approach - convert this!
    result = []
    for x in range(20):
        if x % 3 == 0:
            result.append(x**2)
    return result

def task_1_pythonic():
    """Your Pythonic solution here"""
    return [x**2 for x in range(20) if x % 3 == 0]

# =============================================================================
# TASK 2: String Formatting
# =============================================================================

def task_2_non_pythonic(products):
    """Convert this to use f-strings"""
    # Non-Pythonic approach - convert this!
    messages = []
    for product in products:
        message = ( "Product: " + product["name"] + ", Price: $" + str(product["price"])
                    + ", Stock: " + str(product["stock"]) )
        messages.append(message)
    return messages

def task_2_pythonic(products):
    """Your Pythonic solution here"""
    return [f"Product: {product['name']}, Price: ${product['price']}, Stock: {product['stock']}" 
            for product in products]

# Test data for task 2
sample_products = [
    {"name": "Laptop", "price": 999.99, "stock": 5},
    {"name": "Mouse", "price": 25.50, "stock": 50},
    {"name": "Keyboard", "price": 75.00, "stock": 20}
]

# =============================================================================
# TASK 3: Dictionary Operations
# =============================================================================

def task_3_non_pythonic(user_data, default_role="guest"):
    """Convert this to use .get() method"""
    # Non-Pythonic approach - convert this!
    result = {}
    for user_id, info in user_data.items():
        if "role" in info:
            role = info["role"]
        else:
            role = default_role
        result[user_id] = role
    return result

def task_3_pythonic(user_data, default_role="guest"):
    """Your Pythonic solution here"""
    return {user_id: info.get("role", default_role) for user_id, info in user_data.items()}

# Test data for task 3
sample_users = {
    "user1": {"name": "Alice", "role": "admin"},
    "user2": {"name": "Bob"},
    "user3": {"name": "Charlie", "role": "moderator"},
    "user4": {"name": "Diana"}
}

# =============================================================================
# TASK 4: Parallel Iteration
# =============================================================================

def task_4_non_pythonic(names, scores, subjects):
    """Convert this to use zip()"""
    # Non-Pythonic approach - convert this!
    results = []
    for i in range(len(names)):
        result = names[i] + " scored " + str(scores[i]) + " in " + subjects[i]
        results.append(result)
    return results

def task_4_pythonic(names, scores, subjects):
    """Your Pythonic solution here"""
    return [f"{name} scored {score} in {subject}" 
            for name, score, subject in zip(names, scores, subjects, strict=True)]

# Test data for task 4
student_names = ["Alice", "Bob", "Charlie"]
test_scores = [95, 87, 92]
test_subjects = ["Math", "Science", "English"]

# =============================================================================
# TASK 5: Any/All Functions
# =============================================================================

def task_5a_non_pythonic(numbers):
    """Check if any number is negative - convert to use any()"""
    # Non-Pythonic approach - convert this!
    has_negative = False
    for num in numbers:
        if num < 0:
            has_negative = True
            break
    return has_negative

def task_5a_pythonic(numbers):
    """Your Pythonic solution here"""
    return any(num < 0 for num in numbers)

def task_5b_non_pythonic(words):
    """Check if all words are longer than 3 characters - convert to use all()"""
    # Non-Pythonic approach - convert this!
    all_long = True
    for word in words:
        if len(word) <= 3:
            all_long = False
            break
    return all_long

def task_5b_pythonic(words):
    """Your Pythonic solution here"""
    return all(len(word) > 3 for word in words)

# =============================================================================
# TASK 6: Enumerate and Unpacking
# =============================================================================

def task_6_non_pythonic(items):
    """Convert this to use enumerate() and unpacking"""
    # Non-Pythonic approach - convert this!
    indexed_items = []
    for i in range(len(items)):
        item = items[i]
        # Assume each item is a tuple like ("product", price, quantity)
        name = item[0]
        price = item[1]
        quantity = item[2]
        indexed_items.append(f"Item {i}: {name} costs ${price} (qty: {quantity})")
    return indexed_items

def task_6_pythonic(items):
    """Your Pythonic solution here"""
    return [f"Item {i}: {name} costs ${price} (qty: {quantity})"
            for i, (name, price, quantity) in enumerate(items)]

# Test data for task 6
inventory_items = [
    ("Laptop", 999.99, 5),
    ("Mouse", 25.50, 50),
    ("Keyboard", 75.00, 20)
]

# =============================================================================
# TASK 7: Generator Expression
# =============================================================================

def task_7_non_pythonic(data):
    """Convert this to use generator expression for memory efficiency"""
    # Non-Pythonic approach - convert this!
    # This creates a large list in memory
    squared_evens = [x**2 for x in data if x % 2 == 0]
    return sum(squared_evens)

def task_7_pythonic(data):
    """Your Pythonic solution here"""
    return sum(x**2 for x in data if x % 2 == 0)

# =============================================================================
# TASK 8: Context Manager Challenge
# =============================================================================

def task_8_non_pythonic(filename):
    """Convert this to use context manager"""
    # Non-Pythonic approach - convert this!
    try:
        file = open(filename)
        lines = file.readlines()
        file.close()
        
        # Count non-empty lines
        count = 0
        for line in lines:
            if line.strip():
                count += 1
        return count
    except FileNotFoundError:
        return 0

def task_8_pythonic(filename):
    """Your Pythonic solution here"""
    try:
        with open(filename) as file:
            return sum(1 for line in file if line.strip())
    except FileNotFoundError:
        return 0

# =============================================================================
# TASK 9: Chained Comparisons and Truthiness
# =============================================================================

def task_9_non_pythonic(scores):
    """Convert this to use chained comparisons and truthiness"""
    # Non-Pythonic approach - convert this!
    passing_scores = []
    for score in scores:
        if score >= 60 and score <= 100 and len(passing_scores) >= 0:
            passing_scores.append(score)
    
    if len(passing_scores) > 0:
        return True
    else:
        return False

def task_9_pythonic(scores):
    """Your Pythonic solution here"""
    passing_scores = [score for score in scores if 60 <= score <= 100]
    return bool(passing_scores)

# =============================================================================
# BONUS TASK: Combine Multiple Concepts
# =============================================================================

def bonus_task_non_pythonic(student_data):
    """
    Convert this complex function to be fully Pythonic
    This function should:
    1. Filter students with grades >= 70
    2. Create a summary string for each
    3. Return only if there are results
    """
    # Non-Pythonic approach - convert this!
    results = []
    for i in range(len(student_data)):
        student = student_data[i]
        name = student[0]
        grades = student[1]
        subject = student[2]
        
        total = 0
        count = 0
        for grade in grades:
            total += grade
            count += 1
        
        if count > 0:
            average = total / count
        else:
            average = 0
            
        if average >= 70 and average <= 100:
            message = ("Student " + str(i+1) + ": " + name + " has average "
                        + str(round(average, 1)) + " in " + subject)
            results.append(message)
    
    if len(results) > 0:
        return results
    else:
        return []

def bonus_task_pythonic(student_data):
    """Your Pythonic solution here"""
    results = [
        f"Student {i+1}: {name} has average {round(sum(grades)/len(grades), 1)} in {subject}"
        for i, (name, grades, subject) in enumerate(student_data)
        if grades and 70 <= sum(grades)/len(grades) <= 100
    ]
    return results

# Test data for bonus task
sample_students = [
    ("Alice", [85, 92, 78, 96], "Math"),
    ("Bob", [65, 70, 68, 72], "Science"),
    ("Charlie", [95, 88, 92, 90], "English"),
    ("Diana", [45, 55, 50, 60], "History")
]

# =============================================================================
# TEST FUNCTIONS - Run these to check your solutions
# =============================================================================

def test_all_tasks():
    """Test all implemented functions"""
    print("Testing Pythonic implementations...\n")
    
    # Test Task 1
    print("Task 1 - List Comprehensions:")
    result = task_1_pythonic()
    expected = [0, 9, 36, 81, 144, 225, 324]
    print(f"Result: {result}")
    print(f"Correct: {result == expected}\n")
    
    # Test Task 2
    print("Task 2 - String Formatting:")
    result = task_2_pythonic(sample_products)
    print(f"Result: {result[0]}")  # Show first result
    print(f"Contains proper f-string format: {'Product: Laptop, Price: $999.99, Stock: 5' in result[0]}\n") # noqa : E501
    
    # Test Task 3
    print("Task 3 - Dictionary Operations:")
    result = task_3_pythonic(sample_users)
    expected = {"user1": "admin", "user2": "guest", "user3": "moderator", "user4": "guest"}
    print(f"Result: {result}")
    print(f"Correct: {result == expected}\n")
    
    # Test Task 4
    print("Task 4 - Parallel Iteration:")
    result = task_4_pythonic(student_names, test_scores, test_subjects)
    print(f"Result: {result}")
    print(f"Correct format: {'Alice scored 95 in Math' in result}\n")
    
    # Test Task 5a
    print("Task 5a - Any function:")
    result = task_5a_pythonic([1, 2, -3, 4])
    print(f"Result: {result}")
    print(f"Correct: {result}\n")
    
    # Test Task 5b
    print("Task 5b - All function:")
    result = task_5b_pythonic(["hello", "world", "python"])
    print(f"Result: {result}")
    print(f"Correct: {result}\n")
    
    # Test Task 6
    print("Task 6 - Enumerate and Unpacking:")
    result = task_6_pythonic(inventory_items)
    print(f"Result: {result[0]}")  # Show first result
    print(f"Correct format: {'Item 0: Laptop costs $999.99 (qty: 5)' in result[0]}\n")
    
    # Test Task 7
    print("Task 7 - Generator Expression:")
    result = task_7_pythonic([1, 2, 3, 4, 5, 6])
    expected = 2**2 + 4**2 + 6**2  # 4 + 16 + 36 = 56
    print(f"Result: {result}")
    print(f"Correct: {result == expected}\n")
    
    # Test Task 8 (create a test file first)
    print("Task 8 - Context Manager:")
    test_filename = "test_file.txt"
    try:
        with open(test_filename, 'w') as f:
            f.write("Line 1\n\nLine 3\n")
        result = task_8_pythonic(test_filename)
        print(f"Result: {result}")
        print(f"Correct: {result == 2}\n")
        # Clean up
        import os
        os.remove(test_filename)
    except Exception as e:
        print(f"Error testing file operations: {e}\n")
    
    # Test Task 9
    print("Task 9 - Chained Comparisons and Truthiness:")
    result = task_9_pythonic([45, 75, 95, 55, 85])
    print(f"Result: {result}")
    print(f"Correct: {result}\n")
    
    # Test Bonus Task
    print("Bonus Task - Multiple Concepts:")
    result = bonus_task_pythonic(sample_students)
    print(f"Result: {result}")
    print(f"Contains Alice: {'Alice' in str(result)}")
    print(f"Contains Charlie: {'Charlie' in str(result)}")
    print(f"Excludes Diana: {'Diana' not in str(result)}\n")

if __name__ == "__main__":
    test_all_tasks()