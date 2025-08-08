# Conditionals in Python

# Conditional Statements
# Conditional statements allow you to execute code based on certain conditions. 
# The most common conditional statements are `if`, `elif`, and `else`.

# Basic If Statement
x = 10
if x > 5:
    print("x is greater than 5")  # This block executes if the condition is true

# If-Else Statement
y = 3
if y > 5:
    print("y is greater than 5")
else:
    print("y is not greater than 5")

# If-Elif-Else Statement
z = 7
if z > 10:
    print("z is greater than 10")
elif z > 5:
    print("z is greater than 5 but less than or equal to 10")
else:
    print("z is 5 or less")

# Nested If Statements
a = 15
if a > 10:
    print("a is greater than 10")
    if a > 20:
        print("a is also greater than 20")
    else:
        print("a is not greater than 20")   

# Conditional Expressions (Ternary Operator)
b = 5
result = "b is greater than 3" if b > 3 else "b is not greater than 3"
print(result)  # This will print "b is greater than 3"

# Chained Comparisons
c = 8
if 5 < c < 10:
    print("c is between 5 and 10")  # This checks if c is greater than 5 and less than 10

# Logical Operators in Conditions
d = 12
if d > 10 and d < 15:
    print("d is between 10 and 15")  # This checks if d is both greater than 10 and less than 15    

# Using 'in' for Membership Testing
fruits = ["apple", "banana", "cherry"]
if "banana" in fruits:
    print("Banana is in the list of fruits")  # This checks if "banana" is an element in the fruits list

# Using 'not' for Negation
if "orange" not in fruits:
    print("Orange is not in the list of fruits")  # This checks if "orange" is not an element in the fruits list

# Short-Circuit Evaluation
# Python uses short-circuit evaluation for logical operators.
# This means that if the first condition is false, the second condition is not evaluated.
x = 0
if x != 0 and (10 / x) > 1:  # The second condition is not evaluated because x is 0
    print("This will not print because x is 0") 

# Using 'is' for Identity Testing
a = [1, 2, 3]
b = a
if a is b:
    print("a and b refer to the same object")  # This checks if a and b are the same object in memory

# Using 'is not' for Identity Negation
c = [1, 2, 3]
if a is not c:
    print("a and c do not refer to the same object")  # This checks if a and c are different objects in memory  
