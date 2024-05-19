# Define a string variable
phrase = "Good Morning!"

# Print the string
print(phrase)

# Access individual characters in the string
first_character = phrase[0]
print("The first character is:", first_character)

# Find the length of the string
length = len(phrase)
print("The length of the string is:", length)

# Get user input for the name
name = input("Enter your name: ")

# Concatenate (combine) two strings
personal_greeting = "Good Morning, " + name + "!"
print(personal_greeting)

# Use string methods
uppercase_greeting = personal_greeting.upper()
print("Uppercase greeting:", uppercase_greeting)

# Check if a substring is in the string
contains_evening = "Evening" in personal_greeting
print("Does the greeting contain 'Evening'? ", contains_evening)

# Replace part of the string
updated_phrase = phrase.replace("Morning", "Afternoon")
print("Updated phrase:", updated_phrase)