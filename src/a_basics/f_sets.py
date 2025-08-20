# Sets in Python

# Set Creation
empty_set = set()  # Empty set
numbers_set = {1, 2, 3, 4, 5}  # Set of integers
fruits_set = {"apple", "banana", "cherry"}  # Set of strings
mixed_set = {1, "two", 3.0, True}  # Mixed data types

# Set Methods
fruits_set.add("orange")  # Add an element to the set
fruits_set.remove("banana")  # Remove an element by value (raises KeyError if not found)
fruits_set.discard("banana")  # Remove an element by value (does not raise an error if not found)
popped_fruit = fruits_set.pop()  # Remove and return an arbitrary element
fruits_set.clear()  # Remove all elements from the set
fruits_set_copy = fruits_set.copy()  # Create a shallow copy of the set
fruits_set.update({"kiwi", "mango"})  # Add multiple elements to the set
fruits_set.intersection_update({"apple", "kiwi"})  # Keep only elements found in both sets
fruits_set.difference_update({"kiwi"})  # Remove elements found in another set
fruits_set.symmetric_difference_update({"banana", "kiwi"})  # Keep elements that are in either set but not both

# Set Operations
set_a = {1, 2, 3}
set_b = {3, 4, 5}
union_set = set_a | set_b  # Union of two sets
intersection_set = set_a & set_b  # Intersection of two sets
difference_set = set_a - set_b  # Difference of two sets
symmetric_difference_set = set_a ^ set_b  # Symmetric difference of two sets

# Set Length
length_of_fruits_set = len(fruits_set)  # Get the length of the set

# Set Iteration
for fruit in fruits_set:  # Iterating through a set
    print(fruit)  # Print each fruit

# Set Comparison
set1 = {1, 2, 3}
set2 = {1, 2, 4}
is_equal = set1 == set2  # Check if sets are equal  

is_subset = set1 < set2  # Check if set1 is a subset of set2
is_superset = set1 > set2  # Check if set1 is a superset of set2
is_disjoint = set1.isdisjoint(set2)  # Check if sets have no elements in common

# Set Comprehensions
squared_numbers_set = {x ** 2 for x in range(1, 6)}  # Create a set with squared values
even_numbers_set = {x for x in range(1, 11) if x % 2 == 0}  # Create a set with even numbers

# Frozen Sets
# Immutable version of a set
frozen_set = frozenset([1, 2, 3, 4, 5])  # Create a frozen set
frozen_set_length = len(frozen_set)  # Get the length of the frozen set
frozen_set_elements = list(frozen_set)  # Convert frozen set to a list

#Nested Sets
nested_set = {frozenset({1, 2}), frozenset({3, 4})}  # Set of frozen sets
for item in nested_set:  # Iterating through a nested set
    print(item)  # Print each frozen set
# Note: Sets are unordered collections, so the order of elements may vary.
# Use sets when you need unique elements and do not care about order.


