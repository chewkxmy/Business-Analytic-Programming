# Create a list of strings
cities = ["New York", "London", "Tokyo", "Sydney", "Paris"]

# Print the list
print("Original list:", cities)

# Access elements by index
first_city = cities[0]
print("The first city is:", first_city)

# Slice the list to get a subset
subset_cities = cities[1:3]
print("Subset of the list:", subset_cities)

# Modify an element in the list
cities[2] = "Berlin"
print("Modified list:", cities)

# Append an element to the end of the list
cities.append("Moscow")
print("List after appending 'Moscow':", cities)

# Remove an element by value
cities.remove("London")
print("List after removing 'London':", cities)

# Find the index of an element
index_of_sydney = cities.index("Sydney")
print("Index of 'Sydney':", index_of_sydney)

# Check if an element is in the list
contains_paris = "Paris" in cities
print("Does the list contain 'Paris'?", contains_paris)

# Sort the list
cities.sort()
print("Sorted list:", cities)

# Reverse the list
cities.reverse()
print("Reversed list:", cities)

# Create a list of numbers
numbers = [8, 3, 7, 1, 9]

# Print the first element of the numbers list
print("The first number is:", numbers[0])