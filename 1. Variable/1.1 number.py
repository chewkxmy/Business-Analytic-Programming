# Get user input for integer and floating-point numbers
x = float(input("Enter a number (x): "))
y = float(input("Enter another number (y): "))
z = int(input("Enter an integer (z): "))

# Perform arithmetic operations
sum_xy = x + y
difference_xz = x - z
product_yz = y * z
quotient_xy = x / y
modulus_xy = x % y
exponentiation_xy = x ** y

# Print the results
print("x + y =", sum_xy)
print("x - z =", difference_xz)
print("y * z =", product_yz)
print("x / y =", quotient_xy)
print("x % y =", modulus_xy)
print("x ** y =", exponentiation_xy)

# Use built-in functions for numerical operations
absolute_z = abs(z)
rounded_y = round(y)
max_value = max(x, y, z)
min_value = min(x, y, z)

print("Absolute value of z:", absolute_z)
print("Rounded value of y:", rounded_y)
print("Max value:", max_value)
print("Min value:", min_value)

# Import the math module for more advanced math operations
import math

square_root_x = math.sqrt(x)
logarithm_base_10_x = math.log10(x)
factorial_z = math.factorial(abs(z))

print("Square root of x:", square_root_x)
print("Logarithm base 10 of x:", logarithm_base_10_x)
print(f"Factorial of |z| ({abs(z)}):", factorial_z)