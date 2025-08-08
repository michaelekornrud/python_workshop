# Lists in Python

# List Creation
empty_list = []  # Empty list
numbers = [1, 2, 3, 4, 5]  # List of integers
fruits = ["apple", "banana", "cherry"]  # List of strings
mixed_list = [1, "two", 3.0, True]  # Mixed data types

# List Indexing and Slicing
first_fruit = fruits[0]  # First element
last_fruit = fruits[-1]  # Last element
sublist = fruits[1:3]  # Slicing from index 1 to 3 (exclusive)

# List Methods
fruits.append("orange")  # Add an element to the end
fruits.insert(1, "kiwi")  # Insert an element at index 1
fruits.remove("banana")  # Remove an element by value
popped_fruit = fruits.pop()  # Remove and return the last element
fruits.sort()  # Sort the list in place
fruits.reverse()  # Reverse the list in place
fruits_count = fruits.count("apple")  # Count occurrences of an element
fruits_index = fruits.index("cherry")  # Find the index of an element

# Additional List Operations
fruits.clear()  # Remove all elements from the list
fruits_copy = fruits.copy()  # Create a shallow copy of the list
combined_list = numbers + fruits  # Concatenate two lists
repeated_list = numbers * 2  # Repeat the list

# List Length
length_of_fruits = len(fruits)  # Get the length of the fruits list

# List Comprehensions
squared_numbers = [x ** 2 for x in numbers]  # Create a new list with squared values
even_numbers = [x for x in numbers if x % 2 == 0]  # Filter even numbers

# Nested Lists
nested_list = [[1, 2], [3, 4], [5, 6]]  # List of lists
first_element_of_nested = nested_list[0][0]  # Accessing an element in a nested list    

# List Unpacking
a, b, *rest = numbers  # Unpacking with rest
first_number, second_number, *remaining_numbers = numbers  # Unpacking with remaining numbers  
third_number, *other_numbers = numbers  # Unpacking with other numbers

# List Iteration
for fruit in fruits:  # Iterating through a list
    print(fruit)  # Print each fruit    

# List Comprehensions with Conditions
filtered_fruits = [fruit for fruit in fruits if "a" in fruit]  # Filter fruits containing 'a'
squared_numbers = [x ** 2 for x in numbers if x > 2]  # Square numbers greater than 2

# List as Stack and Queue
stack = []  # Using list as a stack
stack.append(1)  # Push to stack
stack.append(2)  # Push to stack
last_item = stack.pop()  # Pop from stack   

queue = []  # Using list as a queue
queue.append(1)  # Enqueue
queue.append(2)  # Enqueue
first_item = queue.pop(0)  # Dequeue (removing the first item)

# List Comprehensions with Nested Loops
matrix = [[1, 2], [3, 4], [5, 6]]
flattened = [item for sublist in matrix for item in sublist]  # Flatten a nested list   

# List Comprehensions with Enumerate
indexed_fruits = [(index, fruit) for index, fruit in enumerate(fruits)]

# List Comprehensions with Zip
numbers = [1, 2, 3]
letters = ['a', 'b', 'c']
zipped = [(num, letter) for num, letter in zip(numbers, letters)]

# List Comprehensions with Conditional Expressions
squared_or_zero = [x ** 2 if x > 0 else 0 for x in numbers]  # Square positive numbers, zero otherwise

# List Comprehensions with Multiple Conditions
filtered_numbers = [x for x in numbers if x > 1 and x < 4]  # Filter numbers between 1 and 4
