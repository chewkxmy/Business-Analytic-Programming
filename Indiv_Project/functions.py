import pandas as pd
import os

def verify_user(ic_number, password):
    """
    Verify the user's credentials by checking if the IC number is 12 digits long 
    and if the password matches the last 4 digits of the IC number.
    """
    return len(ic_number) == 12 and password == ic_number[-4:]

def calculate_tax(income, tax_relief):
    """
    Calculate the tax payable based on the Malaysian tax rates for the current year.
    This is a simplified version. Real tax calculations would be more complex.
    """
    taxable_income = income - tax_relief
    tax_payable = 0
    
    # Simplified progressive tax rate
    if taxable_income <= 5000:
        tax_payable = 0
    elif taxable_income <= 20000:
        tax_payable = (taxable_income - 5000) * 0.01
    elif taxable_income <= 35000:
        tax_payable = 15000 * 0.01 + (taxable_income - 20000) * 0.03
    elif taxable_income <= 50000:
        tax_payable = 15000 * 0.01 + 15000 * 0.03 + (taxable_income - 35000) * 0.6
    elif tax_payable <=70000
        tax_payable = 15000 * 0.01 + 15000 * 0.03 + 15000 * 0.6 + (taxable_income - 50000) * 0.11
    else:
        tax_payable = 15000 * 0.01 + 15000 * 0.03 + 15000 * 0.6 + 20000 * 0.11 + (taxable_income - 70000) * 0.19
    
    return max(0, tax_payable)

def save_to_csv(data, filename):
    """
    Save the user's data (IC number, income, tax relief, and tax payable) to a CSV file.
    If the file doesn't exist, create a new file with a header row.
    If the file exists, append the new data to the existing file.
    """
    file_exists = os.path.isfile(filename)
    
    df = pd.DataFrame([data], columns=["IC Number", "Income", "Tax Relief", "Tax Payable"])
    
    if file_exists:
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

def read_from_csv(filename):
    """
    Read data from the CSV file and return a pandas DataFrame containing the data.
    If the file doesn't exist, return None.
    """
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return None