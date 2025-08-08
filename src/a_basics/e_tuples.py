# Tuples in Python

# Tuple Creation
empty_tuple = ()  # Empty tuple
single_element_tuple = (42,)  # Single element tuple (note the comma)
numbers = (1, 2, 3, 4, 5)  # Tuple of integers
fruits = ("apple", "banana", "cherry")  # Tuple of strings
mixed_tuple = (1, "two", 3.0, True)  # Mixed data types

# Tuple Indexing and Slicing
first_fruit = fruits[0]  # First element
last_fruit = fruits[-1]  # Last element
subtuple = fruits[1:3]  # Slicing from index 1 to 3 (exclusive)

# Tuple Methods
# Tuples are immutable, so they have fewer methods than lists
fruits_count = fruits.count("apple")  # Count occurrences of an element
fruits_index = fruits.index("cherry")  # Find the index of an element   

# Tuple Length
length_of_fruits = len(fruits)  # Get the length of the fruits tuple

# Tuple Unpacking
a, b, c = numbers  # Unpacking a tuple into variables
first_number, second_number, *remaining_numbers = numbers  # Unpacking with remaining numbers

# Nested Tuples
nested_tuple = ((1, 2), (3, 4), (5, 6))  # Tuple of tuples
first_element_of_nested = nested_tuple[0][0]  # Accessing an element in a nested tuple  

# Tuple Iteration
for fruit in fruits:  # Iterating through a tuple
    print(fruit)  # Print each fruit    

# Tuple as Immutable Lists
# Tuples can be used where immutability is required, such as keys in dictionaries
coordinates = (10.0, 20.0)  # Tuple representing coordinates    

# Tuple Comparison
tuple1 = (1, 2, 3)
tuple2 = (1, 2, 4)
is_equal = tuple1 == tuple2  # Check if tuples are equal
is_greater = tuple1 > tuple2  # Lexicographical comparison

