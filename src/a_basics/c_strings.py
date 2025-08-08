# Strings in Python

# String Creation
greeting = "Hello, World!"  # Using double quotes
quote = 'Python is fun!'  # Using single quotes

multiline_string = """This is a
multiline string that spans
multiple lines."""  # Using triple quotes

single_line_string = '''This is also a
multiline string, but with single quotes.'''  # Using triple single quotes

""" This is a docstring-style comment.
It can be used to describe the purpose of a module, class, or function.
It can also be used for multiline comments."""

''' This is another docstring-style comment.
It can be used to describe the purpose of a module, class, or function.
It can also be used for multiline comments.'''

# String Concatenation
first_part = "Hello"
second_part = "World"
full_greeting = first_part + ", " + second_part + "!"  # Using +

# String Formatting
name = "Alice"
age = 30
formatted_string = f"My name is {name} and I am {age} years old."  # Using f-string

# String Methods
text = "  Python Programming  "
stripped_text = text.strip()  # Remove leading and trailing whitespace
upper_text = text.upper()  # Convert to uppercase
lower_text = text.lower()  # Convert to lowercase
title_text = text.title()  # Convert to title case
replaced_text = text.replace("Python", "Java")  # Replace substring
split_text = text.split()  # Split into a list of words
joined_text = " ".join(split_text)  # Join list of words into a string  

# String Indexing and Slicing
sample_string = "Hello, World!"
first_char = sample_string[0]  # First character
last_char = sample_string[-1]  # Last character
substring = sample_string[0:5]  # Slicing from index 0 to 5 (exclusive)
reversed_string = sample_string[::-1]  # Reverse the string

# String Length
length_of_string = len(sample_string)  # Get the length of the string

# String Comparison
string1 = "apple"
string2 = "banana"
is_equal = string1 == string2  # Check if strings are equal
is_greater = string1 > string2  # Lexicographical comparison    

# Escape Characters
escaped_string = "He said, \"Python is great!\""  # Using backslash to escape quotes
newline_string = "First line.\nSecond line."  # Newline character
tabbed_string = "First line.\n\tSecond line."  # Tab character for indentation