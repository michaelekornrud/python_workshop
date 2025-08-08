# Dictionary in Python
# A dictionary is a collection of key-value pairs.

# Dictionary Creation
empty_dict = {}  # Empty dictionary

# Example of creating a dictionary
my_dict = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}  # A sample dictionary

# Dictionary Methods
my_dict["email"] = "alice@example.com" # Add a new key-value pair
my_dict["phone"] = "123-456-7890" # Add another key-value pair
my_dict.pop("age") # Remove a key-value pair by key
my_dict.update({"country": "USA"}) # Update dictionary with new key-value pairs
my_dict.clear() # Remove all key-value pairs from the dictionary

# Dictionary Access
name = my_dict.get("name")  # Access value by key (returns None if key does not exist)
age = my_dict.get("age", "Not specified")  # Access with default value if key does not exist
city = my_dict["city"]  # Access value by key (raises KeyError if key does not exist)

# Dictionary Iteration
for key, value in my_dict.items():  # Iterate through key-value pairs
    print(f"{key}: {value}")  # Print each key-value pair 

# Dictionary Length
length_of_dict = len(my_dict)  # Get the number of key-value pairs in the dictionary

# Dictionary Keys and Values
keys = my_dict.keys()  # Get all keys in the dictionary
values = my_dict.values()  # Get all values in the dictionary

