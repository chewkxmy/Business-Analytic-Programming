# Sets are unordered collections of unique elements in Python.
# They are often used when you need to work with a collection of
# items where duplicates are not allowed, or when you need to perform set operations like union and intersection.

# Create a set
vehicles = {"car", "bike", "truck", "bus"}

# Add an element to the set
vehicles.add("scooter")

# Remove an element from the set
vehicles.remove("truck")

# Check if an element is in the set
contains_bike = "bike" in vehicles
contains_boat = "boat" in vehicles

# Iterate through the set
print("Vehicles:")
for vehicle in vehicles:
    print(vehicle)

# Create another set
electric_vehicles = {"electric car", "electric bike", "scooter"}

# Perform set operations
union_vehicles_electric = vehicles.union(electric_vehicles)
intersection_vehicles_electric = vehicles.intersection(electric_vehicles)
difference_vehicles_electric = vehicles.difference(electric_vehicles)

# Print the results
print("Contains 'bike'? ", contains_bike)
print("Contains 'boat'? ", contains_boat)
print("Union of vehicles and electric vehicles:", union_vehicles_electric)
print("Intersection of vehicles and electric vehicles:", intersection_vehicles_electric)
print("Difference between vehicles and electric vehicles:", difference_vehicles_electric)