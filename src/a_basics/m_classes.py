# Classes in Python

class Person:
    """A simple class to represent a person."""
    
    def __init__(self, name, age):
        """Initialize the person with a name and age."""
        self.name = name
        self.age = age

    def greet(self):
        """Method to greet the person."""
        return f"Hello, my name is {self.name} and I am {self.age} years old."
    

class Child(Person):
    """A class to represent a child, inheriting from Person."""
    
    def __init__(self, name, age, school):
        """Initialize the child with a name, age, and school."""
        super().__init__(name, age)  # Call the parent class constructor 
        self.school = school

    def greet(self):
        """Method to greet the child."""
        return f"Hello, my name is {self.name}, I am {self.age} years old, and I go to {self.school}."

# Example of using the Person class
if __name__ == "__main__":
    person1 = Person("Alice", 30)
    print(person1.greet())  # Prints: Hello, my name is Alice and I am 30 years old.

    print("\n") # For prettier output
    
    # Example of using the Child class
    child1 = Child("Bob", 10, "Elementary School")
    print(child1.greet())  # Prints: Hello, my name is Bob, I am 10 years old, and I go to Elementary School.