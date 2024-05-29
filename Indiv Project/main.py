import pandas as pd
from functions import verify_user, calculate_tax, save_to_csv, read_from_csv

def main():
    FILENAME = "tax_records.csv"
    print("Welcome to the Malaysian Tax Input Program")

    registered = False
    user_id = input("Enter your user ID: ")
    ic_number = input("Enter your IC number (12 digits): ")

    if verify_user(ic_number, ic_number[-4:]):
        print("You are registered.")
        registered = True
    else:
        print("Registering new user.")
        password = input("Set your password (last 4 digits of IC): ")
        if verify_user(ic_number, password):
            registered = True
        else:
            print("Registration failed. Exiting program.")
            return

    if registered:
        password = input("Enter your password (last 4 digits of IC): ")
        if not verify_user(ic_number, password):
            print("Invalid credentials. Exiting program.")
            return

        try:
            income = float(input("Enter your annual income: "))
            tax_relief = float(input("Enter your total tax relief amount: "))
        except ValueError:
            print("Invalid input for income or tax relief. Exiting program.")
            return

        tax_payable = calculate_tax(income, tax_relief)
        print(f"Your calculated tax payable is: RM{tax_payable:.2f}")

        user_data = [ic_number, income, tax_relief, tax_payable]
        save_to_csv(user_data, FILENAME)
        print("Your data has been saved.")

        # Displaying all records
        records = read_from_csv(FILENAME)
        if records is not None:
            print("\nCurrent Tax Records:")
            print(records)
        else:
            print("No tax records found.")

if __name__ == "__main__":
    main()