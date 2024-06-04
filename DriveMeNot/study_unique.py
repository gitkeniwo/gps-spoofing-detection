import pandas as pd

# Sample DataFrame
data = pd.DataFrame({
    'A': [1, 2, 2, 4, 5, 2, 3, 1],
    'B': [10, 20, 270, 40, 50, 20, 30, 10],
    'C': [100, 200, 200, 400, 500, 600, 700, 800]
})

print("Original DataFrame:\n", data)
unique_A =  data.drop_duplicates(subset=['A'])
print("Unique values in column 'A':", unique_A)