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
    # TODO: Implement using list comprehension
    pass

# =============================================================================
# TASK 2: String Formatting
# =============================================================================

def task_2_non_pythonic(products):
    """Convert this to use f-strings"""
    # Non-Pythonic approach - convert this!
    messages = []
    for product in products:
        message = ("Product: " + product["name"] + ", Price: $" + str(product["price"]) + 
                   ", Stock: " + str(product["stock"]))
        messages.append(message)
    return messages

def task_2_pythonic(products):
    """Your Pythonic solution here"""
    # TODO: Implement using f-strings and list comprehension
    pass

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
    # TODO: Implement using .get() method and dict comprehension
    pass

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
    # TODO: Implement using zip() and f-strings
    pass

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
    # TODO: Implement using any()
    pass

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
    # TODO: Implement using all()
    pass

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
    # TODO: Implement using enumerate() and unpacking
    pass

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
    # TODO: Implement using generator expression
    pass

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
    # TODO: Implement using context manager and Pythonic counting
    pass

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
    # TODO: Implement using chained comparisons and truthiness
    pass

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
            message = ("Student " + str(i+1) + ": " + name + " has average " +
                        str(round(average, 1)) + " in " + subject)
            results.append(message)
    
    if len(results) > 0:
        return results
    else:
        return []

def bonus_task_pythonic(student_data):
    """Your Pythonic solution here"""
    # TODO: Implement using multiple Pythonic concepts
    pass

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
    try:
        result = task_1_pythonic()
        expected = [0, 9, 36, 81, 144, 225, 324]
        print(f"Result: {result}")
        print(f"Correct: {result == expected}\n")
    except Exception:
        print("Not implemented yet\n")
    
    # Test Task 2
    print("Task 2 - String Formatting:")
    try:
        result = task_2_pythonic(sample_products)
        print(f"Result: {result[:1]}...")  # Show first result
        print(f"Contains f-string format: {'Product: Laptop, Price: $999.99, Stock: 5' in str(result)}\n") # noqa : E501
    except Exception:
        print("Not implemented yet\n")
    
    # Test Task 3
    print("Task 3 - Dictionary Operations:")
    try:
        result = task_3_pythonic(sample_users)
        expected = {"user1": "admin", "user2": "guest", "user3": "moderator", "user4": "guest"}
        print(f"Result: {result}")
        print(f"Correct: {result == expected}\n")
    except Exception:
        print("Not implemented yet\n")
    
    # Test Task 4
    print("Task 4 - Parallel Iteration:")
    try:
        result = task_4_pythonic(student_names, test_scores, test_subjects)
        print(f"Result: {result}")
        print(f"Correct format: {'Alice scored 95 in Math' in result}\n")
    except Exception:
        print("Not implemented yet\n")
    
    # Test Task 5a
    print("Task 5a - Any function:")
    try:
        result = task_5a_pythonic([1, 2, -3, 4])
        print(f"Result: {result}")
        print(f"Correct: {result}\n")
    except Exception:
        print("Not implemented yet\n")
    
    # Test Task 5b
    print("Task 5b - All function:")
    try:
        result = task_5b_pythonic(["hello", "world", "python"])
        print(f"Result: {result}")
        print(f"Correct: {result}\n")
    except Exception:
        print("Not implemented yet\n")

if __name__ == "__main__":
    test_all_tasks()