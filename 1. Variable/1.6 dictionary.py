# Create a dictionary of book information
book = {
    "title": "The Great Gatsby",
    "author": "F. Scott Fitzgerald",
    "year_published": 1925,
    "genres": {
        "primary": "Fiction",
        "secondary": "Classic"
    }
}

# Access dictionary values by keys
book_title = book["title"]
book_author = book["author"]

# Modify dictionary values
book["year_published"] = 1926
book["genres"]["primary"] = "Literary Fiction"

# Add a new key-value pair
book["pages"] = 218

# Check if a key exists in the dictionary
has_author = "author" in book
has_isbn = "isbn" in book

# Get the list of keys and values
keys = book.keys()
values = book.values()

# Iterate through the dictionary
print("Book Information:")
for key, value in book.items():
    print(f"{key}: {value}")

# Remove a key-value pair
del book["genres"]

# Print the updated dictionary
print("\nBook Information after removing 'genres':")
for key, value in book.items():
    print(f"{key}: {value}")