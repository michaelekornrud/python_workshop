# Loops in Python

# Loops allow you to execute a block of code repeatedly based on a condition or over a sequence.
# For Loop
# A for loop iterates over a sequence (like a list, tuple, or string).
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)  # Print each fruit in the list

# While Loop
# A while loop continues to execute as long as a condition is true.
count = 0
while count < 5:
    print("Count is:", count)  # Print the current count
    count += 1  # Increment the count

# Loop Control Statements
# Break: Exit the loop prematurely
for i in range(5):
    if i == 3:
        break  # Exit the loop when i is 3
    print(i)  # Print i

# Continue: Skip the current iteration and continue with the next
for i in range(5):
    if i == 2:
        continue  # Skip the iteration when i is 2
    print(i)  # Print i

# Pass: Do nothing, used as a placeholder
for i in range(5):
    if i == 2:
        pass  # Do nothing when i is 2
    print(i)  # Print i

# Nested Loops
# You can have loops inside loops.
for i in range(3):  # Outer loop
    for j in range(2):  # Inner loop
        print(f"i: {i}, j: {j}")  # Print the current values of i and j

# Looping with Enumerate
# Enumerate adds a counter to an iterable and returns it as an enumerate object.
for index, fruit in enumerate(fruits):
    print(f"Index: {index}, Fruit: {fruit}")  # Print the index and fruit   

# Looping with Zip
# Zip combines two or more iterables into tuples.
numbers = [1, 2, 3]
letters = ['a', 'b', 'c']
for number, letter in zip(numbers, letters):
    print(f"Number: {number}, Letter: {letter}")  # Print the number and letter

# Looping with List Comprehensions
# List comprehensions provide a concise way to create lists.
squared_numbers = [x ** 2 for x in range(5)]  # Create a list of squared numbers
print(squared_numbers)  # Print the list of squared numbers

# Looping with Generators
# Generators allow you to iterate over a sequence without storing the entire sequence in memory.
def generate_numbers(n):
    for i in range(n):
        yield i  # Yield the next number in the sequence    

for number in generate_numbers(5):
    print(number)  # Print each generated number
