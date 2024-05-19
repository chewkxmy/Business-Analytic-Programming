# Create a tuple
colors = ("red", "green", "blue", "yellow")

# Access elements by index
first_color = colors[0]
second_color = colors[1]

# Iterate through the tuple
print("Colors:")
for color in colors:
    print(color)

# Check if an element is in the tuple
contains_blue = "blue" in colors

# Find the length of the tuple
num_colors = len(colors)

# Concatenate two tuples
more_colors = ("purple", "orange")
all_colors = colors + more_colors

# Nested tuple
nested_tuple = ("primary", ("secondary", "tertiary"))

# Print the results
print("First color:", first_color)
print("Second color:", second_color)
print("Does it contain 'blue'? ", contains_blue)
print("Number of colors:", num_colors)
print("All colors:", all_colors)
print("Nested tuple:", nested_tuple)