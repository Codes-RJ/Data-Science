#========================================== Introduction ==========================================#

"""_________________<INSTALLATION & SETUP>_________________"""

# What is pandas?
# Pandas is a high-level data analysis and manipulation library built on top of NumPy.

# How it helps:
# - Tabular data handling (rows/columns) with powerful indexing and selection.
# - Fast groupby, aggregation, reshaping, and time-series operations.
# - Rich I/O support: CSV, Excel, SQL databases, JSON, Parquet, and more.

# Installation (run these in terminal / command prompt, NOT inside Python):
#   pip install pandas

# Standard import convention
import pandas as pd

print("Pandas version:", pd.__version__)      # Optional: check installed version

"""_________________<CORE OBJECTS: SERIES VS DATAFRAME>_________________"""

# Pandas has two core labeled data structures:
# - Series  : 1D labeled array of data (values + index).
# - DataFrame: 2D labeled table (rows + columns + index).
# Almost everything in pandas builds on these two objects.

# 1. Creating a Series (1D labeled array)
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print("Series s:\n", s)
"""
Series s:
a    10
b    20
c    30
dtype: int64
"""

print("\nValues of s:", s.values)      # Underlying NumPy array of values
print("Index of s:", s.index)         # Labels along the axis
"""
Values of s: [10 20 30]
Index of s: Index(['a', 'b', 'c'], dtype='object')
"""

# 2. Creating a DataFrame (2D labeled table)
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age":  [25,      30,   35],
    "Score": [88.5,   92.0, 79.5]
}
df = pd.DataFrame(data)
print("\nDataFrame df:\n", df)
"""
DataFrame df:
      Name  Age  Score
0    Alice   25   88.5
1      Bob   30   92.0
2  Charlie   35   79.5
"""

print("\nValues of df:\n", df.values)   # 2D NumPy array of values
print("Index of df:", df.index)        # Row labels
print("Columns of df:", df.columns)    # Column labels
"""
Values of df:
[['Alice' 25 88.5]
 ['Bob' 30 92.0]
 ['Charlie' 35 79.5]]
Index of df: RangeIndex(start=0, stop=3, step=1)
Columns of df: Index(['Name', 'Age', 'Score'], dtype='object')
"""

# 3. Series vs DataFrame: single column and row selection
#    A single column of a DataFrame is a Series.
age_col = df["Age"]
print("\nType of df['Age']:", type(age_col))
print("df['Age'] as Series:\n", age_col)
"""
Type of df['Age']: <class 'pandas.core.series.Series'>
df['Age'] as Series:
0    25
1    30
2    35
Name: Age, dtype: int64
"""

# 4. Alignment behavior (index-aware operations)
#    When adding two Series or DataFrames, pandas aligns on index/labels.
s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
s2 = pd.Series([10, 20, 30], index=["b", "c", "d"])

print("\nSeries s1:\n", s1)
print("\nSeries s2:\n", s2)

print("\nAligned sum (s1 + s2):\n", s1 + s2)
"""
Aligned sum (s1 + s2):
a     NaN   # 'a' only in s1
b    12.0   # 2 + 10
c    23.0   # 3 + 20
d     NaN   # 'd' only in s2
dtype: float64
"""

# 5. Converting between Series and DataFrame
# Series -> DataFrame (as a column)
s_as_df = s.to_frame(name="values")
print("\nSeries s converted to DataFrame:\n", s_as_df)
"""
Series s converted to DataFrame:
   values
a      10
b      20
c      30
"""

# DataFrame -> Series (one column)
name_series = df["Name"]
print("\nDataFrame column 'Name' as Series:\n", name_series)
"""
DataFrame column 'Name' as Series:
0      Alice
1        Bob
2    Charlie
Name: Name, dtype: object
"""

#========================================== Data Ingestion & Export ==========================================#

"""_________________<DATAFRAME & SERIES CREATION>_________________"""

import pandas as pd
import numpy as np

# 1. Creating a Series

# From a Python list (default integer index)
s1 = pd.Series([10, 20, 30])
print("Series from list (default index):\n", s1)
"""
Series from list (default index):
0    10
1    20
2    30
dtype: int64
"""

# From a list with custom index labels
s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])
print("\nSeries from list with custom index:\n", s2)
"""
Series from list with custom index:
a    10
b    20
c    30
dtype: int64
"""

# From a dictionary (keys become index, values become data)
s3 = pd.Series({"x": 1, "y": 2, "z": 3})
print("\nSeries from dictionary:\n", s3)
"""
Series from dictionary:
x    1
y    2
z    3
dtype: int64
"""

# From a scalar (value repeated for each index)
s4 = pd.Series(5, index=["p", "q", "r"])
print("\nSeries from scalar:\n", s4)
"""
Series from scalar:
p    5
q    5
r    5
dtype: int64
"""

# 2. Creating a DataFrame

# From a dictionary of lists (keys -> column names)
data_dict = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age":  [25,      30,   35],
    "Score": [88.5,   92.0, 79.5]
}
df1 = pd.DataFrame(data_dict)
print("\nDataFrame from dict of lists:\n", df1)
"""
DataFrame from dict of lists:
      Name  Age  Score
0    Alice   25   88.5
1      Bob   30   92.0
2  Charlie   35   79.5
"""

# From a list of dictionaries (each dict is a row)
rows = [
    {"Name": "Alice", "Age": 25},
    {"Name": "Bob",   "Age": 30, "City": "Delhi"},
    {"Name": "Cara",  "City": "Mumbai"}
]
df2 = pd.DataFrame(rows)
print("\nDataFrame from list of dicts:\n", df2)
"""
DataFrame from list of dicts:
   Name   Age    City
0  Alice  25.0    NaN
1    Bob  30.0  Delhi
2   Cara   NaN  Mumbai
"""

# From a 2D list (like a matrix), with specified columns
data_2d = [
    [1, "Alice", 88.5],
    [2, "Bob",   92.0],
    [3, "Cara",  79.5]
]
df3 = pd.DataFrame(data_2d, columns=["ID", "Name", "Score"])
print("\nDataFrame from 2D list:\n", df3)
"""
DataFrame from 2D list:
   ID   Name  Score
0   1  Alice   88.5
1   2    Bob   92.0
2   3   Cara   79.5
"""

# From a NumPy array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
df4 = pd.DataFrame(arr, columns=["A", "B", "C"])
print("\nDataFrame from NumPy array:\n", df4)
"""
DataFrame from NumPy array:
   A  B  C
0  1  2  3
1  4  5  6
"""

# 3. Creating empty or minimally structured DataFrames

# Empty DataFrame
df_empty = pd.DataFrame()
print("\nEmpty DataFrame:\n", df_empty)
"""
Empty DataFrame:
Columns: []
Index: []
"""

# DataFrame with specified columns but no rows yet
df_cols_only = pd.DataFrame(columns=["Name", "Age", "Score"])
print("\nDataFrame with only column names:\n", df_cols_only)
"""
DataFrame with only column names:
Empty DataFrame
Columns: [Name, Age, Score]
Index: []
"""

# DataFrame with specified index but no columns yet
df_index_only = pd.DataFrame(index=[0, 1, 2])
print("\nDataFrame with only index:\n", df_index_only)
"""
DataFrame with only index:
Empty DataFrame
Columns: []
Index: [0, 1, 2]
"""

"""_________________<IMPORTING TABULAR DATA (CSV, EXCEL, SQL, WEB)>_________________"""

# This section shows how to read tabular data into a DataFrame from:
# - CSV files
# - Excel files
# - SQL databases
# - HTML tables on web pages
# The idea is always: "source" -> pd.read_xxx(...) -> DataFrame.

# Common options:
# - sep      : delimiter (default is ',')
# - header   : row number to use as column names
# - names    : custom column names
# - usecols  : subset of columns to read
# - nrows    : limit number of rows (useful for large files)

# Example with options:
df_csv_subset = pd.read_csv("data.csv", usecols=["Name", "Age"], nrows=100)
"""
Reads only 'Name' and 'Age' columns and only the first 100 rows.
"""

# 2. Importing from Excel files

# Example: Importing from Excel"
df_excel = pd.read_excel("data.xlsx", sheet_name="Sheet1")
"""
Typical usage:
df_excel = pd.read_excel("data.xlsx")            # Default: first sheet
df_excel = pd.read_excel("data.xlsx", sheet_name="Sheet1")
"""

# 3. Importing from SQL databases
# Use a connection/engine (e.g., via SQLAlchemy) and read a table or a query.
from sqlalchemy import create_engine

# Example SQLite engine (works as a self-contained demo DB file):
engine = create_engine("sqlite:///example.db")

# Example: Importing full table from SQL"
df_sql_table = pd.read_sql("people_demo", con=engine)
"""
Reads an entire table named 'people_demo' into a DataFrame.
"""

# Example: Importing result of a SQL query"
query = "SELECT id, name FROM people_demo WHERE id > 1"
df_sql_query = pd.read_sql(query, con=engine)
"""
Executes the SQL query and returns the result as a DataFrame.
"""

# 4. Importing HTML tables from web pages
# Some web pages contain <table> elements; pandas can try to parse them.

# Example: Importing HTML tables from a web page"
url = "https://example.com/some-page-with-table.html"
tables = pd.read_html(url)
"""
pd.read_html(url) returns a list of DataFrames (one per detected HTML table).
You can select the one you want, e.g., tables[0].
"""

# 5. Importing from JSON
# JSON is common for APIs and web data; if it's record-oriented, it maps nicely to rows.
# Example: Importing from JSON"
df_json = pd.read_json("data.json")
"""
For many APIs, you might first save the JSON response, or pass a URL directly:
df_json = pd.read_json("https://api.example.com/data")
"""

"""_________________<EXPORTING TABULAR DATA (CSV, EXCEL, SQL)>_________________"""

import pandas as pd
from sqlalchemy import create_engine

# This section shows how to save a DataFrame to:
# - CSV files
# - Excel files
# - SQL databases
# The idea is: DataFrame -> df.to_xxx(...) -> external file / table.

df = pd.DataFrame(
    {
        "Name": ["Alice", "Bob", "Cara"],
        "Age":  [25,      30,    35],
        "Score": [88.5,   92.0,  79.5],
    }
)
print("Base DataFrame:\n", df)

# 1. Exporting to CSV
# CSV is a plain-text format; easy to inspect and share.

# Exporting to CSV"
df.to_csv("data_out.csv", index=False)
"""
Typical usage:
df.to_csv("data_out.csv", index=False)    # Save without the index column
df.to_csv("data_out.csv", index=True)     # Save with index as first column
"""

# 2. Exporting to Excel
# Requires an Excel writer engine (like openpyxl or xlsxwriter) installed.

# Exporting to Excel"
df.to_excel("data_out.xlsx", sheet_name="Sheet1", index=False)
"""
Typical usage:
df.to_excel("data_out.xlsx", index=False)                      # Default sheet name
df.to_excel("data_out.xlsx", sheet_name="Scores", index=False) # Custom sheet name
"""

# 3. Exporting to SQL databases with to_sql

from sqlalchemy import create_engine
engine = create_engine("sqlite:///example_export.db")
# Exporting to SQL table"
df.to_sql("scores", con=engine, if_exists="replace", index=False)
"""
Common patterns:
# Create table or fail if it exists:
df.to_sql("scores", con=engine, if_exists="fail", index=False)

# Drop existing table and recreate it:
df.to_sql("scores", con=engine, if_exists="replace", index=False)

# Append new rows to existing table:
df.to_sql("scores", con=engine, if_exists="append", index=False)
"""

"""_________________<UPDATING & DELETING DATA RECORDS VIA PANDAS>_________________"""

import pandas as pd

# - Update/delete records in a DataFrame using pandas operations.
# - Optionally push those changes back to external storage (CSV/Excel/SQL).

# Sample DataFrame
df = pd.DataFrame(
    {
        "id":   [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Cara", "Dan"],
        "age":  [25, 30, 22, 28],
    }
)
print("Original DataFrame:\n", df)
"""
Original DataFrame:
   id   name  age
0   1  Alice   25
1   2    Bob   30
2   3   Cara   22
3   4    Dan   28
"""

# 1. UPDATING RECORDS IN A DATAFRAME

# Example: increase age by 1 where name == "Bob"
df.loc[df["name"] == "Bob", "age"] = df.loc[df["name"] == "Bob", "age"] + 1
print("\nAfter incrementing Bob's age by 1:\n", df)
"""
Uses boolean mask + .loc to update specific cells.
"""

# Example: set age = 0 for all rows where age < 25
df.loc[df["age"] < 25, "age"] = 0
print("\nAfter setting age=0 where age < 25:\n", df)
"""
Vectorized conditional update across multiple rows at once.
"""

# 2. DELETING RECORDS (ROWS) IN A DATAFRAME

# Example: drop rows where name == "Dan"
mask = df["name"] == "Dan"
df_dropped = df[~mask]          # keep rows where mask is False
print("\nAfter deleting rows where name == 'Dan':\n", df_dropped)
"""
Common pattern: filter with a boolean condition to keep only desired rows.
"""

# Alternative: use drop() with an index
df_drop_idx = df.drop(index=df[df["age"] == 0].index)
print("\nAfter dropping rows where age == 0 (using .drop):\n", df_drop_idx)
"""
df[df["age"] == 0].index selects matching row indices, then drop() removes them.
"""

# 3. SYNCING UPDATES/DELETES BACK TO EXTERNAL STORAGE (CSV / SQL)

# 3a. CSV / Excel: overwrite with updated DataFrame
# Overwriting CSV/Excel with updated DataFrame (pattern):
df_dropped.to_csv("people_updated.csv", index=False)
df_dropped.to_excel("people_updated.xlsx", sheet_name="people", index=False)
"""
Typical flow:
- Load:     df = pd.read_csv("people.csv")
- Modify:   update/delete rows via .loc, boolean masks, .drop
- Save:     df.to_csv("people.csv", index=False)   # overwrite original file
"""

# 3b. SQL: simple pattern = overwrite or append

from sqlalchemy import create_engine

# Create SQLite engine (example)
engine = create_engine("sqlite:///people_update.sqlite")

# Write original df to SQL table
df.to_sql("people", con=engine, if_exists="replace", index=False)

# Now suppose df_dropped is our "updated" table state.
df_dropped.to_sql("people", con=engine, if_exists="replace", index=False)
print("\n# SQL table 'people' replaced with updated DataFrame (overwrite pattern).")
"""
Basic overwrite pattern:
- Read table into df using pd.read_sql(...)
- Modify df using pandas (updates/deletes)
- Write back with if_exists='replace' to reflect the new state
For more granular row-level updates/deletes, you typically:
- Use pandas to compute what should change
- Then execute custom SQL UPDATE/DELETE statements via the DB API / SQLAlchemy.
"""

#========================================== DataFrame Exploration & Manipulation ==========================================#

"""_________________<BASIC ATTRIBUTES & METHODS>_________________"""

import pandas as pd

# Sample DataFrame for this section
data = {
    "Name":  ["Alice", "Bob", "Cara", "Dan", "Eve"],
    "Age":   [25,      30,     22,    28,   35],
    "Score": [88.5,    92.0,   76.0,  85.5, 90.0],
}
df = pd.DataFrame(data)
print("Sample DataFrame:\n", df)
"""
Sample DataFrame:
   Name  Age  Score
0  Alice   25   88.5
1    Bob   30   92.0
2   Cara   22   76.0
3    Dan   28   85.5
4    Eve   35   90.0
"""

# 1. Shape, size, dimensions
print("\nShape (rows, columns):", df.shape)                     # (number_of_rows, number_of_columns)
print("Number of elements (size):", df.size)                    # rows * columns
print("Number of dimensions (ndim):", df.ndim)
"""
Example output:
Shape (rows, columns): (5, 3)
Number of elements (size): 15
Number of dimensions (ndim): 2
"""


# 2. Index, columns, dtypes
print("\nIndex labels:", df.index)
print("Column labels:", df.columns)
print("Data types:\n", df.dtypes)
"""
Index labels: RangeIndex(start=0, stop=5, step=1)
Column labels: Index(['Name', 'Age', 'Score'], dtype='object')
Data types:
Name     object
Age       int64
Score   float64
dtype: object
"""

# 3. Quick row previews: head(), tail(), sample()

print("\nFirst 3 rows (head):\n", df.head(3))
print("\nLast 2 rows (tail):\n", df.tail(2))
print("\nRandom 2 rows (sample):\n", df.sample(2, random_state=42))
"""
head(n):  first n rows (default n=5)
tail(n):  last n rows (default n=5)
sample(n): random n rows (use random_state for reproducibility)
"""

# 4. Info and basic memory overview

print("\nDataFrame info():")
df.info()
"""
info() prints:
- Index range
- Column names and counts of non-null values
- Data types per column
- Approximate memory usage
"""


# 5. Descriptive statistics: describe()

print("\nDescriptive statistics (numeric columns):\n", df.describe())
"""
describe() gives count, mean, std, min, quartiles, and max for numeric columns.
Use df.describe(include='object') for string/categorical summaries,
or df.describe(include='all') for a mixed summary.
"""


# 6. Memory usage per column (optional but useful)

print("\nMemory usage per column (bytes):\n", df.memory_usage(deep=True))
"""
memory_usage(deep=True) shows how much memory each column uses,
which is useful when optimizing large DataFrames.
"""

"""_________________<INDEXING, SELECTION & SLICING>_________________"""

import pandas as pd

# Indexing, selection, and slicing let you pick specific rows, columns, or sub-tables.
# Core tools:
# - Column access: df["col"], df[["col1", "col2"]]
# - Label-based:   .loc[row_labels, col_labels]
# - Position-based:.iloc[row_positions, col_positions]
# - Boolean masks and conditions

# Sample DataFrame for this section
data = {
    "Name":  ["Alice", "Bob", "Cara", "Dan", "Eve"],
    "Age":   [25,      30,     22,    28,   35],
    "Score": [88.5,    92.0,   76.0,  85.5, 90.0],
}
df = pd.DataFrame(data, index=["a", "b", "c", "d", "e"])
print("Sample DataFrame (with custom index):\n", df)
"""
Sample DataFrame (with custom index):
   Name  Age  Score
a  Alice   25   88.5
b    Bob   30   92.0
c   Cara   22   76.0
d    Dan   28   85.5
e    Eve   35   90.0
"""

# 1. Column selection

# Single column -> Series
name_col = df["Name"]
print("\nSingle column df['Name']:\n", name_col)
"""
Single column df['Name']:
a    Alice
b      Bob
c     Cara
d      Dan
e      Eve
Name: Name, dtype: object
"""

# Multiple columns -> DataFrame
sub_cols = df[["Name", "Score"]]
print("\nMultiple columns df[['Name', 'Score']]:\n", sub_cols)
"""
Multiple columns df[['Name', 'Score']]:
   Name  Score
a  Alice   88.5
b    Bob   92.0
c   Cara   76.0
d    Dan   85.5
e    Eve   90.0
"""

# 2. Label-based selection with .loc

# Single row by label
row_b = df.loc["b"]
print("\nRow 'b' using .loc:\n", row_b)
"""
Row 'b' using .loc:
Name      Bob
Age        30
Score    92.0
Name: b, dtype: object
"""

# Row slice by labels (inclusive end)
rows_b_to_d = df.loc["b":"d"]
print("\nRows 'b' to 'd' using .loc:\n", rows_b_to_d)
"""
Rows 'b' to 'd' using .loc:
   Name  Age  Score
b   Bob   30   92.0
c  Cara   22   76.0
d   Dan   28   85.5
"""

# Specific rows and columns by label
subset_loc = df.loc[["b", "d"], ["Name", "Score"]]
print("\n.loc with specific rows and columns:\n", subset_loc)
"""
.loc with specific rows and columns:
   Name  Score
b   Bob   92.0
d   Dan   85.5
"""

# 3. Position-based selection with .iloc

# Single row by integer position
row_0 = df.iloc[0]
print("\nRow at position 0 using .iloc:\n", row_0)
"""
Row at position 0 using .iloc:
Name     Alice
Age         25
Score     88.5
Name: a, dtype: object
"""

# Row slice by position (end-exclusive)
rows_1_to_3 = df.iloc[1:4]
print("\nRows positions 1 to 3 using .iloc:\n", rows_1_to_3)
"""
Rows positions 1 to 3 using .iloc:
   Name  Age  Score
b   Bob   30   92.0
c  Cara   22   76.0
d   Dan   28   85.5
"""

# Specific rows and columns by position
subset_iloc = df.iloc[[0, 2, 4], [0, 2]]   # rows 0,2,4 and columns 0 (Name), 2 (Score)
print("\n.iloc with specific row/col positions:\n", subset_iloc)
"""
.iloc with specific row/col positions:
   Name  Score
a  Alice   88.5
c   Cara   76.0
e    Eve   90.0
"""

# 4. Boolean indexing (conditional selection)

# Filter rows where Age > 25
mask_age = df["Age"] > 25
print("\nBoolean mask (Age > 25):\n", mask_age)
"""
Boolean mask (Age > 25):
a    False
b     True
c    False
d     True
e     True
Name: Age, dtype: bool
"""

older_than_25 = df[mask_age]
print("\nRows where Age > 25:\n", older_than_25)
"""
Rows where Age > 25:
   Name  Age  Score
b   Bob   30   92.0
d   Dan   28   85.5
e   Eve   35   90.0
"""

# Combined conditions (use & and | with parentheses)
high_scorers = df[(df["Age"] > 25) & (df["Score"] >= 90)]
print("\nRows where Age > 25 AND Score >= 90:\n", high_scorers)
"""
Rows where Age > 25 AND Score >= 90:
   Name  Age  Score
b   Bob   30   92.0
e   Eve   35   90.0
"""

# 5. Slicing with the index (Series-like behavior)

# Using the index order for slicing (label-based via .loc)
slice_by_label = df.loc["a":"c"]   # inclusive of 'c'
print("\nSlice df.loc['a':'c']:\n", slice_by_label)
"""
Slice df.loc['a':'c']:
   Name  Age  Score
a  Alice   25   88.5
b    Bob   30   92.0
c   Cara   22   76.0
"""

# Using integer slices (position-based via .iloc)
slice_by_pos = df.iloc[1:4]        # end-exclusive (rows 1,2,3)
print("\nSlice df.iloc[1:4]:\n", slice_by_pos)
"""
Slice df.iloc[1:4]:
   Name  Age  Score
b   Bob   30   92.0
c  Cara   22   76.0
d   Dan   28   85.5
"""

"""_________________<ADDING, MODIFYING & DROPPING COLUMNS/ROWS>_________________"""

import pandas as pd

# Sample DataFrame for this section
data = {
    "Name":  ["Alice", "Bob", "Cara"],
    "Age":   [25,      30,    22],
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)
"""
Original DataFrame:
   Name  Age
0  Alice   25
1    Bob   30
2   Cara   22
"""

# 1. ADDING COLUMNS

# a) Add new column from a list/Series (length must match number of rows)
df["Score"] = [88.5, 92.0, 76.0]
print("\nAfter adding 'Score' column:\n", df)
"""
After adding 'Score' column:
   Name  Age  Score
0  Alice   25   88.5
1    Bob   30   92.0
2   Cara   22   76.0
"""

# b) Add column based on existing columns (vectorized operation)
df["Age_Category"] = df["Age"].apply(lambda x: "Adult" if x >= 25 else "Young")
print("\nAfter adding 'Age_Category' based on 'Age':\n", df)
"""
After adding 'Age_Category' based on 'Age':
   Name  Age  Score Age_Category
0  Alice   25   88.5        Adult
1    Bob   30   92.0        Adult
2   Cara   22   76.0        Young
"""

# c) Add the same scalar value for all rows
df["Country"] = "India"
print("\nAfter adding scalar column 'Country':\n", df)
"""
After adding scalar column 'Country':
   Name  Age  Score Age_Category Country
0  Alice   25   88.5        Adult   India
1    Bob   30   92.0        Adult   India
2   Cara   22   76.0        Young   India
"""

# 2. MODIFYING COLUMNS

# a) Modify an entire column (e.g., convert Age from years to months)
df["Age_months"] = df["Age"] * 12
print("\nAfter adding modified column 'Age_months':\n", df)
"""
After adding modified column 'Age_months':
   Name  Age  Score Age_Category Country  Age_months
0  Alice   25   88.5        Adult   India         300
1    Bob   30   92.0        Adult   India         360
2   Cara   22   76.0        Young   India         264
"""

# b) Modify selected rows in a column using .loc
df.loc[df["Name"] == "Bob", "Score"] = 95.0
print("\nAfter updating Bob's Score to 95.0:\n", df)
"""
After updating Bob's Score to 95.0:
   Name  Age  Score Age_Category Country  Age_months
0  Alice   25   88.5        Adult   India         300
1    Bob   30   95.0        Adult   India         360
2   Cara   22   76.0        Young   India         264
"""

# 3. DROPPING COLUMNS

# a) Drop a single column
df_no_country = df.drop(columns=["Country"])
print("\nAfter dropping 'Country' column:\n", df_no_country)
"""
After dropping 'Country' column:
   Name  Age  Score Age_Category  Age_months
0  Alice   25   88.5        Adult         300
1    Bob   30   95.0        Adult         360
2   Cara   22   76.0        Young         264
"""

# b) Drop multiple columns
df_less = df.drop(columns=["Age_months", "Age_Category"])
print("\nAfter dropping 'Age_months' and 'Age_Category':\n", df_less)
"""
After dropping 'Age_months' and 'Age_Category':
   Name  Age  Score Country
0  Alice   25   88.5   India
1    Bob   30   95.0   India
2   Cara   22   76.0   India
"""

# 4. ADDING ROWS

# a) Append a single row using loc with a new index
df.loc[3] = ["Dan", 28, 85.5, "Adult", "India", 28 * 12]
print("\nAfter adding a new row for Dan using .loc:\n", df)
"""
After adding a new row for Dan using .loc:
   Name  Age  Score Age_Category Country  Age_months
0  Alice   25   88.5        Adult   India         300
1    Bob   30   95.0        Adult   India         360
2   Cara   22   76.0        Young   India         264
3    Dan   28   85.5        Adult   India         336
"""

# b) Concatenate another DataFrame vertically (more flexible than deprecated append)
new_rows = pd.DataFrame(
    {
        "Name": ["Eve"],
        "Age":  [35],
        "Score": [90.0],
        "Age_Category": ["Adult"],
        "Country": ["India"],
        "Age_months": [35 * 12],
    }
)
df_extended = pd.concat([df, new_rows], ignore_index=True)
print("\nAfter concatenating a new row for Eve with pd.concat:\n", df_extended)
"""
After concatenating a new row for Eve with pd.concat:
   Name  Age  Score Age_Category Country  Age_months
0  Alice   25   88.5        Adult   India         300
1    Bob   30   95.0        Adult   India         360
2   Cara   22   76.0        Young   India         264
3    Dan   28   85.5        Adult   India         336
4    Eve   35   90.0        Adult   India         420
"""

# 5. DROPPING ROWS

# a) Drop by index labels
df_drop_idx = df_extended.drop(index=[1, 3])                    # drop rows with ind
print("\nAfter dropping rows with index 1 and 3:\n", df_drop_idx)
"""
After dropping rows with index 1 and 3:
   Name  Age  Score Age_Category Country  Age_months
0  Alice   25   88.5        Adult   India         300
2   Cara   22   76.0        Young   India         264
4    Eve   35   90.0        Adult   India         420
"""
# b) Drop rows based on a condition (boolean mask)
df_drop_cond = df_extended[df_extended["Age"] >= 25]            # keep Age >= 25
print("\nAfter dropping rows where Age < 25:\n", df_drop_cond)
"""
After dropping rows where Age < 25:
   Name  Age  Score Age_Category Country  Age_months
0  Alice   25   88.5        Adult   India         300
1    Bob   30   95.0        Adult   India         360
3    Dan   28   85.5        Adult   India         336
4    Eve   35   90.0        Adult   India         420
"""

# 6. Modifying the index
# a) Set a column as the index

df_indexed = df_extended.set_index("Name")
print("\nAfter setting 'Name' as the index:\n", df_indexed)
"""
After setting 'Name' as the index:
       Age  Score Age_Category Country  Age_months
Name
Alice   25   88.5        Adult   India         300
Bob     30   95.0        Adult   India         360
Cara    22   76.0        Young   India         264
Dan     28   85.5        Adult   India         336
Eve     35   90.0        Adult   India         420
"""

# b) Reset index back to default integer index

df_reset = df_indexed.reset_index()
print("\nAfter resetting index back to default:\n", df_reset)
"""
After resetting index back to default:
    Name  Age  Score Age_Category Country  Age_months
0  Alice   25   88.5        Adult   India         300
1    Bob   30   95.0        Adult   India         360
2   Cara   22   76.0        Young   India         264
3    Dan   28   85.5        Adult   India         336
4    Eve   35   90.0        Adult   India         420
"""
"""
       Name  Age  Score Age_Category Country  Age_months
0    Alice   25   88.5        Adult   India         300
1      Bob   30   92.0        Adult   India         360
2    Charlie   22   76.0        Young   India         264
3      Dan   28   85.5        Adult   India         336
4      Eve   35   90.0        Adult   India         420
"""

# 7. Renaming columns
df_renamed = df_extended.rename(columns={"Age": "Age_years", "Score": "Exam_Score"})
print("\nAfter renaming columns 'Age' and 'Score':\n", df_renamed)
"""
After renaming columns 'Age' and 'Score':
       Name  Age_years  Exam_Score Age_Category Country  Age_months
0    Alice         25        88.5        Adult   India         300
1      Bob         30        95.0        Adult   India         360
2    Charlie         22        76.0        Young   India         264
3      Dan         28        85.5        Adult   India         336
4      Eve         35        90.0        Adult   India         420
"""

# 8. Reordering columns
new_order = ["Name", "Country", "Age", "Score", "Age_Category", "Age_months"]
df_reordered = df_extended[new_order]
print("\nAfter reordering columns:\n", df_reordered)
"""
After reordering columns:
       Name Country  Age  Score Age_Category  Age_months
0    Alice   India   25   88.5        Adult         300
1      Bob   India   30   95.0        Adult         360
2    Charlie   India   22   76.0        Young         264
3      Dan   India   28   85.5        Adult         336
4      Eve   India   35   90.0        Adult         420
"""

# 9. Changing data types of columns
df_extended["Age"] = df_extended["Age"].astype(float)
print("\nAfter changing 'Age' column to float type:\n", df_extended.dtypes)
"""
After changing 'Age' column to float type:
Name             object
Age            float64
Score           float64
Age_Category     object
Country          object
Age_months       int64
dtype: object
"""

#========================================== Data Cleaning & Preparation ==========================================#

"""_________________<WORKING WITH DUPLICATES & VALUE COUNTS>_________________"""

import pandas as pd

# Sample DataFrame with intentional duplicates
data = {
    "Name":  ["Alice", "Bob", "Alice", "Cara", "Bob", "Dan"],
    "City":  ["Delhi", "Mumbai", "Delhi", "Delhi", "Pune", "Delhi"],
    "Score": [88,      92,      88,      76,      92,    85],
}
df = pd.DataFrame(data)
print("Original DataFrame (with duplicates):\n", df)
"""
Original DataFrame (with duplicates):
    Name   City  Score
0  Alice  Delhi     88
1    Bob  Mumbai    92
2  Alice  Delhi     88
3   Cara  Delhi     76
4    Bob   Pune     92
5    Dan  Delhi     85
"""

# 1. Detecting duplicate rows

# a) Check which rows are exact duplicates across all columns
dup_mask_all = df.duplicated()
print("\nDuplicate mask (all columns):\n", dup_mask_all)
"""
Duplicate mask (all columns):
0    False
1    False
2     True
3    False
4    False
5    False
dtype: bool
"""

# Show only duplicated rows
dup_rows_all = df[df.duplicated()]
print("\nRows that are duplicates (all columns):\n", dup_rows_all)
"""
Rows that are duplicates (all columns):
    Name   City  Score
2  Alice  Delhi     88
"""

# b) Check duplicates based on a subset of columns (e.g., Name only)
dup_mask_name = df.duplicated(subset=["Name"])
print("\nDuplicate mask based on 'Name':\n", dup_mask_name)
"""
Duplicate mask based on 'Name':
0    False   # First 'Alice'
1    False   # First 'Bob'
2     True   # Second 'Alice'
3    False   # 'Cara'
4     True   # Second 'Bob'
5    False   # 'Dan'
dtype: bool
"""

dup_rows_name = df[df.duplicated(subset=["Name"])]
print("\nRows that are duplicates based on 'Name':\n", dup_rows_name)
"""
Rows that are duplicates based on 'Name':
    Name   City  Score
2  Alice  Delhi     88
4    Bob   Pune     92
"""

# 2. Dropping duplicate rows

# a) Drop exact duplicates (all columns), keeping the first occurrence
df_drop_all = df.drop_duplicates()
print("\nAfter drop_duplicates() on all columns (keep='first'):\n", df_drop_all)
"""
After drop_duplicates() on all columns (keep='first'):
    Name   City  Score
0  Alice  Delhi     88
1    Bob  Mumbai    92
3   Cara  Delhi     76
4    Bob   Pune     92
5    Dan  Delhi     85
"""

# b) Drop duplicates based on 'Name', keep first occurrence
df_drop_name_first = df.drop_duplicates(subset=["Name"], keep="first")
print("\nAfter drop_duplicates(subset=['Name'], keep='first'):\n", df_drop_name_first)
"""
After drop_duplicates(subset=['Name'], keep='first'):
    Name   City  Score
0  Alice  Delhi     88
1    Bob  Mumbai    92
3   Cara  Delhi     76
5    Dan  Delhi     85
"""

# c) Drop duplicates based on 'Name', keep last occurrence
df_drop_name_last = df.drop_duplicates(subset=["Name"], keep="last")
print("\nAfter drop_duplicates(subset=['Name'], keep='last'):\n", df_drop_name_last)
"""
After drop_duplicates(subset=['Name'], keep='last'):
    Name   City  Score
2  Alice  Delhi     88
4    Bob   Pune     92
3   Cara  Delhi     76
5    Dan  Delhi     85
"""

# 3. Value counts (frequency of categories)

# a) Count occurrences of each Name
name_counts = df["Name"].value_counts()
print("\nValue counts for 'Name':\n", name_counts)
"""
Value counts for 'Name':
Alice    2
Bob      2
Cara     1
Dan      1
Name: Name, dtype: int64
"""

# b) Count occurrences of each City
city_counts = df["City"].value_counts()
print("\nValue counts for 'City':\n", city_counts)
"""
Value counts for 'City':
Delhi     4
Mumbai    1
Pune      1
Name: City, dtype: int64
"""

# c) Normalized value counts (relative frequencies)
name_freq = df["Name"].value_counts(normalize=True)
print("\nNormalized value counts for 'Name' (proportions):\n", name_freq)
"""
Normalized value counts for 'Name' (proportions):
Alice    0.333333
Bob      0.333333
Cara     0.166667
Dan      0.166667
Name: Name, dtype: float64
"""

# 4. Handling duplicates + value counts together

# Example: keep only one row per Name, and attach a count column
name_counts_df = df["Name"].value_counts().rename("Name_count")
name_counts_df = name_counts_df.to_frame()
print("\nName counts as a small DataFrame:\n", name_counts_df)
"""
Name counts as a small DataFrame:
       Name_count
Alice           2
Bob             2
Cara            1
Dan             1
"""

# Merge counts back into a de-duplicated DataFrame (keep first occurrence per Name)
df_unique = df.drop_duplicates(subset=["Name"], keep="first")
df_with_counts = df_unique.merge(
    name_counts_df,
    left_on="Name",
    right_index=True,
    how="left",
)
print("\nDe-duplicated DataFrame with per-Name counts:\n", df_with_counts)
"""
De-duplicated DataFrame with per-Name counts:
    Name   City  Score  Name_count
0  Alice  Delhi     88           2
1    Bob  Mumbai    92           2
3   Cara  Delhi     76           1
5    Dan  Delhi     85           1
"""

"""_________________<MISSING DATA HANDLING>_________________"""

import pandas as pd
import numpy as np

# Missing data is usually represented as NaN (Not a Number) in pandas.
# Key tools:
# - Detect:   isna(), notna()
# - Drop:     dropna()
# - Fill:     fillna(), interpolate()
# - Options:  control how rows/columns are treated

# Sample DataFrame with missing values
data = {
    "Name":  ["Alice", "Bob",  "Cara",   "Dan"],
    "Age":   [25,      np.nan, 22,       28],
    "Score": [88.5,    92.0,   np.nan,   85.5],
}
df = pd.DataFrame(data)
print("Original DataFrame with missing values:\n", df)
"""
Original DataFrame with missing values:
   Name   Age  Score
0  Alice  25.0   88.5
1    Bob   NaN   92.0
2   Cara  22.0    NaN
3    Dan  28.0   85.5
"""

# 1. Detecting missing values

print("\nBoolean mask of missing values (isna):\n", df.isna())
"""
isna() (or isnull()) returns True where values are missing:
    Name    Age  Score
0  False  False  False
1  False   True  False
2  False  False   True
3  False  False  False
"""

print("\nCount of missing values per column:\n", df.isna().sum())
"""
Count of missing values per column:
Name     0
Age      1
Score    1
dtype: int64
"""

print("\nRows with any missing values:\n", df[df.isna().any(axis=1)])
"""
any(axis=1) checks if any column in a row is NaN.
"""

# 2. Dropping missing data with dropna()

# a) Drop rows that contain any NaN
df_drop_any = df.dropna()
print("\nAfter dropna() (drop rows with any NaN):\n", df_drop_any)
"""
After dropna():
   Name   Age  Score
0  Alice  25.0   88.5
3    Dan  28.0   85.5
"""

# b) Drop rows only if all entries are NaN
df_all_nan_example = pd.DataFrame(
    {
        "A": [1, np.nan, np.nan],
        "B": [np.nan, np.nan, 3],
    }
)
print("\nExample DataFrame for how='all':\n", df_all_nan_example)
"""
Example:
     A    B
0  1.0  NaN
1  NaN  NaN
2  NaN  3.0
"""

df_drop_all = df_all_nan_example.dropna(how="all")
print("\nAfter dropna(how='all') (drop rows where all values are NaN):\n", df_drop_all)
"""
Row 1 is dropped because all its values are NaN.
"""

# c) Drop columns with too many missing values
df_col_drop = df_all_nan_example.dropna(axis="columns", how="all")
print("\nAfter dropna(axis='columns', how='all'):\n", df_col_drop)
"""
Drops columns that are entirely NaN.
"""

# d) Drop rows with less than a threshold of non-NaN values
df_drop_thresh = df_all_nan_example.dropna(thresh=1)
print("\nAfter dropna(thresh=1) (keep rows with at least 1 non-NaN):\n", df_drop_thresh)
"""
Keeps rows with at least 1 non-NaN value.
"""

# 3. Filling missing data with fillna()

# a) Fill all NaNs with a single value
df_fill_zero = df.fillna(0)
print("\nAfter fillna(0) (fill all NaNs with 0):\n", df_fill_zero)
"""
After fillna(0):
   Name   Age  Score
0  Alice  25.0   88.5
1    Bob   0.0   92.0
2   Cara  22.0    0.0
3    Dan  28.0   85.5
"""

# b) Fill per column using a dict
fill_values = {"Age": df["Age"].mean(), "Score": df["Score"].median()}
df_fill_col = df.fillna(fill_values)
print("\nAfter fillna with column-wise values (mean Age, median Score):\n", df_fill_col)
"""
Use different strategies per column:
- Age   -> mean of known ages
- Score -> median of known scores
"""

# c) Forward fill (propagate last valid value downward)
df_ffill = df.fillna(method="ffill")
print("\nAfter forward fill (ffill):\n", df_ffill)
"""
Forward fill example:
- Bob's Age becomes 25.0 (same as Alice)
- Cara's Score becomes 92.0 (same as Bob)
"""

# d) Backward fill (propagate next valid value upward)
df_bfill = df.fillna(method="bfill")
print("\nAfter backward fill (bfill):\n", df_bfill)
"""
Backward fill example:
- Bob's Age becomes 22.0 (next non-missing Age)
- Cara's Score becomes 85.5 (next non-missing Score)
"""

# 4. Interpolating missing numeric values

# Simple numeric example for interpolation
s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
print("\nOriginal Series for interpolation:\n", s)
"""
Original Series:
0    1.0
1    NaN
2    3.0
3    NaN
4    5.0
dtype: float64
"""

s_interp = s.interpolate()
print("\nAfter interpolate():\n", s_interp)
"""
interpolate() fills NaNs by constructing values between known points.
Result:
0    1.0
1    2.0   # halfway between 1 and 3
2    3.0
3    4.0   # halfway between 3 and 5
4    5.0
dtype: float64
"""

# 5. Controlling which axis and how much to fill

# limit parameter: restrict how many consecutive NaNs to fill
s2 = pd.Series([1.0, np.nan, np.nan, 4.0])
print("\nSeries with consecutive NaNs:\n", s2)
"""
Series with consecutive NaNs:
0    1.0
1    NaN
2    NaN
3    4.0
dtype: float64
"""

s2_ffill_limit = s2.fillna(method="ffill", limit=1)
print("\nForward fill with limit=1:\n", s2_ffill_limit)
"""
Only the first NaN after a valid value is filled:
0    1.0
1    1.0   # filled
2    NaN   # still NaN (limit reached)
3    4.0
dtype: float64
"""

# 6. Combining methods for robust missing data handling
df_combined = df.copy()
df_combined["Age"] = df_combined["Age"].fillna(df_combined["Age"].mean())
df_combined["Score"] = df_combined["Score"].interpolate()
print("\nAfter combined missing data handling:\n", df_combined)
"""
After combined missing data handling:
   Name        Age      Score
0  Alice  25.000000   88.500000
1    Bob  25.000000   92.000000
2   Cara  22.000000   90.250000
3    Dan  28.000000   85.500000
"""

"""_________________<STRING OPERATIONS ON COLUMNS>_________________"""

import pandas as pd

# Pandas provides vectorized string methods via the .str accessor.
# These work on Series of dtype 'object' or 'string' and let you clean, transform, and analyze text data efficiently.

data = {
    "Name": ["alice ", "BOB", "Cara", "dAn"],
    "City": [" new delhi", "MUMBAI ", "Pune", "delhi"],
    "Email": ["alice@example.com", "bob@EXAMPLE.com", None, "dan@example.org"],
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)
"""
Original DataFrame:
      Name        City              Email
0   alice     new delhi  alice@example.com
1      BOB      MUMBAI   bob@EXAMPLE.com
2     Cara        Pune               None
3      dAn       delhi   dan@example.org
"""

# 1. Changing case: lower, upper, title, capitalize

df["Name_lower"] = df["Name"].str.lower()
df["Name_upper"] = df["Name"].str.upper()
df["Name_title"] = df["Name"].str.title()
print("\nCase normalization on 'Name':\n", df[["Name", "Name_lower", "Name_upper", "Name_title"]])
"""
Common case operations:
- .str.lower()
- .str.upper()
- .str.title()
- .str.capitalize()
"""

# 2. Stripping whitespace

df["Name_stripped"] = df["Name"].str.strip()
df["City_stripped"] = df["City"].str.strip()
print("\nAfter stripping leading/trailing spaces:\n", df[["Name", "Name_stripped", "City", "City_stripped"]])
"""
Use .str.strip() to remove leading and trailing whitespace.
Use .str.lstrip() / .str.rstrip() for one side only.
"""

# 3. Contains, startswith, endswith, replace

# a) Check if Email contains a specific domain fragment
df["is_example_domain"] = df["Email"].str.contains("example", case=False, na=False)
print("\nCheck if Email contains 'example' (case-insensitive):\n", df[["Email", "is_example_domain"]])
"""
.str.contains(pattern, case=False, na=False) returns a boolean mask.
"""

# b) startswith / endswith on City
df["city_starts_with_d"] = df["City_stripped"].str.lower().str.startswith("d")
print("\nCities starting with 'd':\n", df[["City_stripped", "city_starts_with_d"]])
"""
Use .str.startswith() and .str.endswith() for prefix/suffix checks.
"""

# c) Replace substrings
df["Email_normalized"] = df["Email"].str.replace("EXAMPLE.com", "example.com", case=False)
print("\nEmail normalized domain:\n", df[["Email", "Email_normalized"]])
"""
.str.replace(old, new, case=...) replaces occurrences of a substring or regex pattern.
"""

# 4. Splitting and extracting parts of strings

# a) Split Email into user and domain
email_split = df["Email"].str.split("@", n=1, expand=True)
df["Email_user"] = email_split[0]
df["Email_domain"] = email_split[1]
print("\nSplit Email into user and domain:\n", df[["Email", "Email_user", "Email_domain"]])
"""
.str.split(sep, n=..., expand=True) splits into columns.
"""

# b) Extract using simple patterns (e.g., domain TLD)
df["Email_tld"] = df["Email_domain"].str.split(".", n=1, expand=True)[1]
print("\nExtract top-level domain from Email:\n", df[["Email_domain", "Email_tld"]])
"""
This example splits 'example.com' into ['example', 'com'] and keeps 'com'.
"""

# 5. Length, counting substrings

df["Name_len"] = df["Name_stripped"].str.len()
print("\nLength of names (without spaces):\n", df[["Name_stripped", "Name_len"]])
"""
.str.len() gives the length of each string.
"""

df["num_a_in_name"] = df["Name_stripped"].str.lower().str.count("a")
print("\nCount of 'a' in each name:\n", df[["Name_stripped", "num_a_in_name"]])
"""
.str.count(substring) counts non-overlapping occurrences in each string.
"""

# 6. Handling missing values in string operations

# Many .str methods return NaN when the original value is NaN.
# You can fill or replace them after operations.

print("\nString operations handle None/NaN gracefully (result may be NaN):")
print(df[["Email", "Email_user", "Email_domain"]])
"""
If you need to ensure strings instead of NaN, you can use .fillna('') or similar:
df["Email_user"] = df["Email_user"].fillna("")
"""

# 7. Chaining string methods
df["City_cleaned"] = df["City"].str.strip().str.lower().str.replace(" ", "_")
print("\nChained string operations on 'City':\n", df[["City", "City_cleaned"]])
"""
Chained operations allow multiple transformations in one line.
"""

"""_________________<WORKING WITH DATES & TIMES>_________________"""

import pandas as pd

# Pandas has strong support for dates and times via:
# - pd.to_datetime() for parsing strings
# - DatetimeIndex for time-aware indexing
# - .dt accessor for vectorized datetime operations
# - Resampling, shifting, rolling windows (time-series)

# 1. Converting strings to datetime with pd.to_datetime

data = {
    "Date_str": ["2024-01-01", "2024-01-02", "2024/01/03", "01-04-2024"],
    "Sales":    [100,           120,           90,           150],
}
df = pd.DataFrame(data)
print("Original DataFrame with date strings:\n", df)
"""
Original DataFrame with date strings:
     Date_str  Sales
0  2024-01-01    100
1  2024-01-02    120
2  2024/01/03     90
3  01-04-2024    150
"""

# Convert to datetime; pandas handles many common formats automatically.
df["Date"] = pd.to_datetime(df["Date_str"], dayfirst=False, errors="coerce")
print("\nAfter pd.to_datetime conversion:\n", df)
"""
After conversion, 'Date' is a datetime64[ns] column usable for time-based operations.
"""

# 2. Setting a DatetimeIndex

df = df.set_index("Date")
print("\nDataFrame with Date as index (DatetimeIndex):\n", df)
"""
DatetimeIndex allows:
- time-based slicing (by year, month, day)
- resampling (e.g., monthly totals)
"""

# Example: select data for a specific date range
subset = df["2024-01-02":"2024-01-03"]
print("\nSlice by date range '2024-01-02' to '2024-01-03':\n", subset)
"""
Uses label-based slicing on the DatetimeIndex.
"""

# 3. Using the .dt accessor for datetime components

# Reset index to get Date back as a column for .dt examples
df_reset = df.reset_index()
print("\nDataFrame for .dt examples:\n", df_reset)
"""
DataFrame for .dt examples:
        Date  Date_str  Sales
0 2024-01-01 2024-01-01    100
1 2024-01-02 2024-01-02    120
2 2024-01-03 2024/01/03     90
3 2024-01-04 01-04-2024    150
"""

df_reset["year"] = df_reset["Date"].dt.year
df_reset["month"] = df_reset["Date"].dt.month
df_reset["day"] = df_reset["Date"].dt.day
df_reset["weekday"] = df_reset["Date"].dt.day_name()
print("\nDatetime components via .dt:\n", df_reset[["Date", "year", "month", "day", "weekday"]])
"""
.dt.year, .dt.month, .dt.day, .dt.day_name() give calendar components for each row.
"""

# 4. Creating date ranges

# Generate a daily DateTimeIndex
date_range = pd.date_range(start="2024-01-01", periods=7, freq="D")
print("\nDate range (daily, 7 periods):\n", date_range)
"""
pd.date_range() is useful for building time indices or simulating time series.
"""

# Create a time series using date_range as index
ts = pd.DataFrame(
    {"value": [10, 12, 11, 15, 14, 13, 16]},
    index=date_range,
)
print("\nTime series with DateTimeIndex:\n", ts)
"""
Time series indexed by dates:
            value
2024-01-01     10
...
2024-01-07     16
"""

# 5. Shifting and lagging

ts["value_shifted_1"] = ts["value"].shift(1)   # shift down by 1 row
ts["value_diff"] = ts["value"] - ts["value_shifted_1"]
print("\nAfter shift and simple difference:\n", ts)
"""
.shift(n) moves data by n periods (NaN introduced where no previous value).
Common in time-series for lag features and returns.
"""

# 6. Resampling time series

# Example: suppose we have hourly data and want daily sums
hourly_index = pd.date_range("2024-01-01", periods=24, freq="H")
hourly_ts = pd.DataFrame({"value": range(24)}, index=hourly_index)
print("\nHourly time series (first few rows):\n", hourly_ts.head())
"""
Hourly time series with 24 values, one per hour.
"""

daily_sum = hourly_ts.resample("D").sum()
print("\nDaily sum using resample('D').sum():\n", daily_sum)
"""
resample('D') groups by calendar day; sum() aggregates within each group.
Common frequencies:
- 'D'  : daily
- 'H'  : hourly
- 'W'  : weekly
- 'M'  : month-end
- 'MS' : month-start
"""

# 7. Handling time zones (basic example)

# Localize naive datetimes to a timezone, then convert
tz_index = pd.date_range("2024-01-01 09:00", periods=3, freq="H")
ts_tz = pd.DataFrame({"value": [1, 2, 3]}, index=tz_index)
ts_tz = ts_tz.tz_localize("Asia/Kolkata")
print("\nTime series localized to Asia/Kolkata:\n", ts_tz)
"""
tz_localize() attaches a timezone to naive timestamps.
"""

ts_tz_utc = ts_tz.tz_convert("UTC")
print("\nConverted to UTC:\n", ts_tz_utc)
"""
tz_convert() converts between time zones while preserving the actual time points.
"""

"""_________________<CATEGORICAL DATA & MEMORY OPTIMIZATION>_________________"""

import pandas as pd
import numpy as np

# Categorical data is useful when a column has repeated labels (e.g., city, product type).
# Converting such columns to 'category' dtype:
# - Reduces memory usage.
# - Speeds up some operations (groupby, value_counts, comparisons).

# 1. Example DataFrame with repeated labels
data = {
    "City":  ["Delhi", "Mumbai", "Delhi", "Pune", "Delhi", "Mumbai", "Chennai", "Chennai"],
    "Segment": ["A", "B", "A", "C", "A", "B", "C", "C"],
    "Sales": np.random.randint(100, 200, size=8),
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)
print("\nOriginal dtypes:\n", df.dtypes)
"""
Original DataFrame:
      City Segment  Sales
0    Delhi       A   ...
1   Mumbai       B   ...
2    Delhi       A   ...
3     Pune       C   ...
4    Delhi       A   ...
5   Mumbai       B   ...
6  Chennai       C   ...
7  Chennai       C   ...

Original dtypes:
City       object
Segment    object
Sales       int64
dtype: object
"""

# 2. Memory usage before converting to category

mem_before = df.memory_usage(deep=True)
print("\nMemory usage BEFORE category conversion (bytes):\n", mem_before)
print("Total memory BEFORE:", mem_before.sum(), "bytes")
"""
memory_usage(deep=True) shows memory per column, including object strings.
"""

# 3. Converting object columns to category

df["City_cat"] = df["City"].astype("category")
df["Segment_cat"] = df["Segment"].astype("category")

print("\nAfter adding categorical versions of City and Segment:")
print(df[["City", "City_cat", "Segment", "Segment_cat"]].head())
print("\nDtypes after adding categorical columns:\n", df.dtypes)
"""
Note: City_cat and Segment_cat now have dtype 'category'.
"""

# 4. Memory usage after converting to category

mem_after = df[["City_cat", "Segment_cat", "Sales"]].memory_usage(deep=True)
print("\nMemory usage AFTER category conversion (subset with categorical columns):\n", mem_after)
print("Total memory AFTER (subset):", mem_after.sum(), "bytes")
"""
Categorical columns usually consume less memory than raw object/string columns,
especially when there are many repeated values.
"""

# 5. Inspecting categories and codes

print("\nCategories for City_cat:", df["City_cat"].cat.categories)
print("Integer codes for City_cat:\n", df["City_cat"].cat.codes)
"""
Categorical columns store:
- .cat.categories : unique labels
- .cat.codes      : integer codes referencing categories
"""

# 6. Useful categorical operations

# a) Reordering categories (e.g., for ordered segments)
df["Segment_cat"] = df["Segment_cat"].cat.reorder_categories(["A", "B", "C"], ordered=True)
print("\nReordered Segment_cat categories (A < B < C):", df["Segment_cat"].cat.categories)
print("Is Segment_cat ordered?", df["Segment_cat"].cat.ordered)
"""
Reordering and setting ordered=True is helpful for comparisons, sorting, and plots.
"""

# b) Sorting by categorical order
df_sorted = df.sort_values("Segment_cat")
print("\nDataFrame sorted by ordered Segment_cat:\n", df_sorted[["Segment_cat", "Sales"]])
"""
sort_values respects the category order when the categorical is ordered.
"""

# 7. Converting back to string/object if needed

df["City_str"] = df["City_cat"].astype(str)
print("\nCity_cat converted back to string:\n", df[["City_cat", "City_str"]].head())
"""
You can always convert a categorical column back to string/object if you need raw text.
"""

# 8. Pattern for memory optimization on a larger DataFrame

print("\n# Example pattern for optimizing memory on many object columns:")
print("""
# Given a DataFrame df with many object columns:
obj_cols = df.select_dtypes(include="object").columns

# Convert high-cardinality columns carefully, low-cardinality columns directly:
for col in obj_cols:
    num_unique = df[col].nunique(dropna=False)
    num_total = len(df[col])
    if num_unique / num_total < 0.5:   # heuristic threshold
        df[col] = df[col].astype("category")
""")
"""
Heuristic:
- If a column has many repeated values (low unique/total ratio), converting to 'category'
  can give good memory savings.
"""

"""_________________<APPLY, MAP & VECTORIZED OPERATIONS>_________________"""

import pandas as pd
import numpy as np

# Pandas encourages vectorized operations (column-wise math) instead of Python loops.
# When you need custom logic, you can use:
# - Series.map()        : element-wise mapping on a single Series
# - Series.apply()      : element-wise function on a Series
# - DataFrame.apply()   : function along rows or columns
# - DataFrame.applymap(): element-wise on all DataFrame entries (less common)

# Sample DataFrame
df = pd.DataFrame(
    {
        "Name":  ["Alice", "Bob", "Cara", "Dan"],
        "Age":   [25, 30, 22, 28],
        "Score": [88.5, 92.0, 76.0, 85.5],
    }
)
print("Original DataFrame:\n", df)
"""
Original DataFrame:
   Name  Age  Score
0  Alice   25   88.5
1    Bob   30   92.0
2   Cara   22   76.0
3    Dan   28   85.5
"""

# 1. Vectorized operations (preferred over loops)

# Example: increase Score by 10% using pure vectorized math
df["Score_boosted"] = df["Score"] * 1.10
print("\nAfter vectorized 10% Score boost:\n", df[["Name", "Score", "Score_boosted"]])
"""
Vectorized operations are fast and concise:
df["Score"] * 1.10 operates on the whole column at once.
"""

# 2. Series.map() — element-wise mapping with dict or function

# a) Map names to shorter labels using a dictionary
name_map = {"Alice": "A", "Bob": "B", "Cara": "C", "Dan": "D"}
df["Name_short"] = df["Name"].map(name_map)
print("\nAfter mapping Name to Name_short with dict and .map:\n", df[["Name", "Name_short"]])
"""
Series.map(dict) replaces each value using the mapping;
values not found in the dict become NaN.
"""

# b) Map with a function (element-wise)
df["Age_plus_5_map"] = df["Age"].map(lambda x: x + 5)
print("\nAge_plus_5 using .map with lambda:\n", df[["Age", "Age_plus_5_map"]])
"""
Series.map(func) applies the function to each element.
"""

# 3. Series.apply() — element-wise, more general than map()

def age_category(age):
    if age < 25:
        return "Young"
    elif age < 30:
        return "Adult"
    else:
        return "Senior"

df["Age_category"] = df["Age"].apply(age_category)
print("\nAge_category using .apply on a Series:\n", df[["Age", "Age_category"]])
"""
Series.apply(func) is similar to map(func), often used for more complex logic.
"""

# 4. DataFrame.apply() — function across rows or columns

# a) Apply along columns (axis=0): e.g., compute range (max - min) per column
col_range = df[["Age", "Score"]].apply(lambda col: col.max() - col.min(), axis=0)
print("\nColumn-wise range (max - min) using DataFrame.apply(axis=0):\n", col_range)
"""
When axis=0 (default), the function receives each column as a Series.
"""

# b) Apply along rows (axis=1): e.g., total points = Age + Score
df["Age_plus_Score"] = df.apply(lambda row: row["Age"] + row["Score"], axis=1)
print("\nRow-wise Age_plus_Score using DataFrame.apply(axis=1):\n", df[["Name", "Age", "Score", "Age_plus_Score"]])
"""
When axis=1, the function receives each row as a Series.
Useful for combining multiple columns into a single computed feature.
"""

# 5. DataFrame.applymap() — element-wise on the whole DataFrame

num_df = df[["Age", "Score"]]
scaled = num_df.applymap(lambda x: round(x / 10, 1))
print("\nElement-wise scaling with applymap on numeric columns:\n", scaled)
"""
applymap(func) applies a function to every element of the DataFrame.
It is less common than vectorized operations, but useful for formatting/cleaning.
"""

# When to use which method?
"""
- Prefer vectorized operations whenever possible:
    df["Score"] * 1.10
    df["Age"] + df["Score"]

- Use Series.map() when:
    * You have a dict or simple function to transform a single column.
    * Example: df["Name"].map({"Alice": "A", "Bob": "B"})

- Use Series.apply() when:
    * You need more complex per-element logic on a single column.
    * Example: df["Age"].apply(age_category)

- Use DataFrame.apply(axis=0/1) when:
    * You need column-wise or row-wise computations combining multiple values.
    * Example: df.apply(lambda col: col.max() - col.min(), axis=0)

- Use DataFrame.applymap() when:
    * You truly need an element-wise transformation over many cells.
    * Example: df.applymap(lambda x: str(x).upper()) for text formatting.
"""

"""_________________<CLEANING & DATA QUALITY UTILITIES>_________________"""

import pandas as pd
import numpy as np

# This section focuses on common data cleaning tasks:
# - Trimming spaces, fixing text casing
# - Handling inconsistent values
# - Dealing with outliers
# - Fixing column names
# - Basic sanity checks (duplicates, missing, ranges)

# Sample dirty DataFrame
data = {
    "Name":  [" alice ", "BOB", "Cara", "dAn", None],
    "Age":   [25, 300, None, 28, 22],          # 300 is suspicious (possible outlier / bad value)
    "City":  ["delhi", " Delhi", "MUMBAI ", "pune", "delhi"],
    "Score": ["88.5", " 92", "NaN", "85.5", ""],  # numeric as strings, missing as "NaN" / ""
}
df = pd.DataFrame(data)
print("Original 'dirty' DataFrame:\n", df)
"""
Original 'dirty' DataFrame:
      Name    Age     City Score
0   alice     25   delhi   88.5
1      BOB   300   Delhi     92
2     Cara   NaN  MUMBAI    NaN
3      dAn    28    pune   85.5
4     None    22   delhi       
"""

# 1. Standardizing text: strip spaces, fix casing

df["Name_clean"] = df["Name"].str.strip().str.title()
df["City_clean"] = df["City"].str.strip().str.title()
print("\nAfter trimming and standardizing Name/City:\n", df[["Name", "Name_clean", "City", "City_clean"]])
"""
- .str.strip() removes leading/trailing spaces.
- .str.title() converts to Title Case (e.g., 'delhi' -> 'Delhi').
"""

# 2. Converting numeric-like strings to real numbers

# Replace common "missing" string markers with NaN, then convert
df["Score_num"] = (
    df["Score"]
    .replace(["", "NaN", "nan", "NULL"], np.nan)
    .astype(float)
)
print("\nScore converted from dirty strings to numeric:\n", df[["Score", "Score_num"]])
"""
.replace([...], np.nan) catches textual missing markers.
.astype(float) converts numeric strings to float dtype.
"""

# 3. Handling impossible / extreme values (basic outlier cleaning)

# Example rule: Age should be between 0 and 120; mark others as NaN
df["Age_clean"] = df["Age"].where((df["Age"] >= 0) & (df["Age"] <= 120), np.nan)
print("\nAge with impossible values set to NaN:\n", df[["Age", "Age_clean"]])
"""
.where(condition, other) keeps values where condition is True, otherwise uses 'other' (NaN here).
"""

# Optionally, fill cleaned Age with a typical value (e.g., median of valid ages)
valid_age_median = df["Age_clean"].median()
df["Age_imputed"] = df["Age_clean"].fillna(valid_age_median)
print("\nAge after simple median imputation:\n", df[["Age_clean", "Age_imputed"]])
"""
Imputation strategy can vary (mean, median, domain-specific value, etc.).
"""

# 4. Cleaning column names

# Example: inconsistent column names
df_columns_demo = pd.DataFrame(columns=[" First Name ", "last-name", "AGE (years)"])
print("\nOriginal messy column names:\n", df_columns_demo.columns)
"""
Index([' First Name ', 'last-name', 'AGE (years)'], dtype='object')
"""

# Clean: strip spaces, lower-case, replace spaces/illegal chars with underscores
df_columns_demo.columns = (
    df_columns_demo.columns
    .str.strip()
    .str.lower()
    .str.replace("[^0-9a-zA-Z]+", "_", regex=True)
)
print("\nCleaned column names:\n", df_columns_demo.columns)
"""
Column name cleaning pattern:
- strip spaces
- to lower
- replace non-alphanumeric sequences with '_'
"""

# 5. Checking basic data quality: missing, duplicates, ranges

# a) Missing values per column
missing_per_col = df[["Name_clean", "Age_imputed", "City_clean", "Score_num"]].isna().sum()
print("\nMissing values per column (cleaned subset):\n", missing_per_col)
"""
Helps you see which columns still have missing data.
"""

# b) Duplicate rows
dups = df.duplicated(subset=["Name_clean", "City_clean"])
print("\nDuplicate mask based on Name_clean + City_clean:\n", dups)
print("\nRows that are duplicates (Name_clean + City_clean):\n", df[dups])
"""
Use .duplicated(...) and .drop_duplicates(...) to inspect and remove duplicated entities.
"""

# c) Range checks with boolean masks

# Example: check Score between 0 and 100
invalid_score_mask = (df["Score_num"] < 0) | (df["Score_num"] > 100)
print("\nRows with Score_num outside [0, 100]:\n", df[invalid_score_mask])
"""
If invalid_score_mask has any True, you may need to correct or drop those rows.
"""

# Combining cleaning steps into a mini pipeline (pattern)
"""
# 1) Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace('[^0-9a-zA-Z]+', '_', regex=True)
)

# 2) Strip text columns and standardize case
text_cols = ['name', 'city']
for col in text_cols:
    df[col] = df[col].astype('string')
    df[col] = df[col].str.strip().str.title()

# 3) Convert numeric-like text to numbers and handle missing
df['score'] = (
    df['score']
    .replace(['', 'NaN', 'nan', 'NULL'], np.nan)
    .astype(float)
)

# 4) Handle out-of-range values
df['age'] = df['age'].where((df['age'] >= 0) & (df['age'] <= 120), np.nan)

# 5) Drop obviously bad rows or duplicates if needed
df = df.drop_duplicates()
"""

"""_________________<JOINING, MERGING & CONCATENATION>_________________"""

import pandas as pd

# This section shows how to combine DataFrames:
# - concat  : stack DataFrames vertically or horizontally
# - merge   : SQL-style joins on keys
# - join    : convenient index-based joins

# Sample DataFrames for examples
df_left = pd.DataFrame(
    {
        "id":   [1, 2, 3],
        "Name": ["Alice", "Bob", "Cara"],
    }
)
df_right = pd.DataFrame(
    {
        "id":    [2, 3, 4],
        "Score": [92.0, 76.0, 88.5],
    }
)
print("Left DataFrame (df_left):\n", df_left)
print("\nRight DataFrame (df_right):\n", df_right)
"""
Left DataFrame (df_left):
   id   Name
0   1  Alice
1   2    Bob
2   3   Cara

Right DataFrame (df_right):
   id  Score
0   2   92.0
1   3   76.0
2   4   88.5
"""

# 1. CONCATENATION (stacking along an axis)

# a) Vertical concatenation (stack rows)
df_top = pd.DataFrame({"id": [5], "Name": ["Dan"]})
df_bottom = pd.DataFrame({"id": [6], "Name": ["Eve"]})

df_vert = pd.concat([df_left, df_top, df_bottom], ignore_index=True)
print("\nVertical concat (rows):\n", df_vert)
"""
pd.concat([...], axis=0) stacks rows (like UNION ALL in SQL, without deduplication).
ignore_index=True resets the index in the result.
"""

# b) Horizontal concatenation (add columns side-by-side)
df_extra = pd.DataFrame(
    {
        "City": ["Delhi", "Mumbai", "Pune"],
    }
)
df_horiz = pd.concat([df_left, df_extra], axis=1)
print("\nHorizontal concat (columns):\n", df_horiz)
"""
pd.concat([...], axis=1) aligns DataFrames by index and adds columns.
"""

# 2. MERGING (SQL-style joins)

# a) Inner join on key 'id' (rows with matching id in both)
df_inner = pd.merge(df_left, df_right, on="id", how="inner")
print("\nInner merge on 'id':\n", df_inner)
"""
Inner join: keeps only keys present in BOTH df_left and df_right.
Result:
   id  Name  Score
1   2   Bob   92.0
2   3  Cara   76.0
"""

# b) Left join (keep all rows from left, match from right when possible)
df_left_join = pd.merge(df_left, df_right, on="id", how="left")
print("\nLeft merge on 'id':\n", df_left_join)
"""
Left join: all rows from df_left are kept; missing matches from df_right become NaN.
Result:
   id   Name  Score
0   1  Alice    NaN
1   2    Bob   92.0
2   3   Cara   76.0
"""

# c) Right join (keep all rows from right)
df_right_join = pd.merge(df_left, df_right, on="id", how="right")
print("\nRight merge on 'id':\n", df_right_join)
"""
Right join: all rows from df_right are kept; missing matches from df_left become NaN.
Result:
   id   Name  Score
0   2    Bob   92.0
1   3   Cara   76.0
2   4    NaN   88.5
"""

# d) Outer join (keep all keys from both sides)
df_outer = pd.merge(df_left, df_right, on="id", how="outer")
print("\nOuter merge on 'id':\n", df_outer)
"""
Outer join: union of keys from both DataFrames.
Missing values are filled with NaN.
Result:
   id   Name  Score
0   1  Alice    NaN
1   2    Bob   92.0
2   3   Cara   76.0
3   4    NaN   88.5
"""

# 3. MERGING ON DIFFERENT KEY NAMES
df_a = pd.DataFrame(
    {
        "user_id": [1, 2, 3],
        "Name":    ["Alice", "Bob", "Cara"],
    }
)
df_b = pd.DataFrame(
    {
        "id":    [2, 3, 4],
        "Score": [92.0, 76.0, 88.5],
    }
)
df_merge_diff_keys = pd.merge(df_a, df_b, left_on="user_id", right_on="id", how="inner")
print("\nMerge with different key names (left_on, right_on):\n", df_merge_diff_keys)
"""
Use left_on and right_on when key column names differ between DataFrames.
"""

# 4. JOINING BY INDEX (.join)

# Prepare DataFrames with index-based keys
df_left_idx = df_left.set_index("id")
df_right_idx = df_right.set_index("id")
print("\nLeft with id as index:\n", df_left_idx)
print("\nRight with id as index:\n", df_right_idx)
"""
Both DataFrames now have 'id' as the index.
"""

df_join = df_left_idx.join(df_right_idx, how="left")
print("\nIndex-based join using .join:\n", df_join)
"""
DataFrame.join() joins on index by default.
Equivalent to a left join on the index.
"""

# 5. Handling overlapping column names
df_left2 = pd.DataFrame(
    {
        "id":   [1, 2, 3],
        "Value": ["L1", "L2", "L3"],
    }
)
df_right2 = pd.DataFrame(
    {
        "id":   [2, 3, 4],
        "Value": ["R2", "R3", "R4"],
    }
)
df_overlap = pd.merge(df_left2, df_right2, on="id", how="outer", suffixes=("_left", "_right"))
print("\nMerge with overlapping column names (using suffixes):\n", df_overlap)
"""
suffixes=("_left", "_right") helps distinguish columns with same name from left and right.
"""

# 6. Combining multiple DataFrames

df1 = pd.DataFrame({"id": [1, 2], "A": ["A1", "A2"]})
df2 = pd.DataFrame({"id": [2, 3], "B": ["B2", "B3"]})
df3 = pd.DataFrame({"id": [1, 3], "C": ["C1", "C3"]})
df_combined = df1.merge(df2, on="id", how="outer").merge(df3, on="id", how="outer")
print("\nCombining multiple DataFrames with successive merges:\n", df_combined)
"""
Combining multiple DataFrames can be done with successive merges.
Result:
   id    A    B    C
0   1   A1  NaN   C1
1   2   A2   B2  NaN
2   3  NaN   B3   C3
"""

#========================================== Data Analysis ==========================================#

"""_________________<DESCRIPTIVE STATISTICS & SUMMARIES>_________________"""

import pandas as pd
import numpy as np

# Descriptive stats help you quickly summarize distributions:
# - describe(), mean(), median(), std(), min(), max()
# - value_counts(), unique(), nunique()
# - corr(), cov()

# Sample DataFrame
np.random.seed(0)
df = pd.DataFrame(
    {
        "Age":   [25, 30, 22, 28, 35],
        "Score": [88.5, 92.0, 76.0, 85.5, 90.0],
        "City":  ["Delhi", "Mumbai", "Delhi", "Pune", "Delhi"],
    }
)
print("Sample DataFrame:\n", df)
"""
Sample DataFrame:
   Age  Score   City
0   25   88.5  Delhi
1   30   92.0 Mumbai
2   22   76.0  Delhi
3   28   85.5   Pune
4   35   90.0  Delhi
"""

# 1. describe(): quick summary of numeric columns

print("\nNumeric summary (describe):\n", df.describe())
"""
describe() returns count, mean, std, min, quartiles, and max for numeric columns.
"""

# 2. Individual stats: mean, median, std, min, max

print("\nAge mean:", df["Age"].mean())
print("Age median:", df["Age"].median())
print("Age std:", df["Age"].std())
print("Age min:", df["Age"].min())
print("Age max:", df["Age"].max())
"""
Single-column summary stats.
"""

print("\nColumn-wise means:\n", df[["Age", "Score"]].mean())
"""
mean() on a DataFrame returns column-wise means by default.
"""

# 3. unique(), nunique(), value_counts() for categorical/label columns

print("\nUnique cities:", df["City"].unique())
print("Number of unique cities:", df["City"].nunique())
print("\nCity value counts:\n", df["City"].value_counts())
"""
unique() -> array of unique values.
nunique() -> count of unique values.
value_counts() -> frequency of each category.
"""

# 4. Correlation and covariance for numeric columns

print("\nCorrelation matrix (Age & Score):\n", df[["Age", "Score"]].corr())
print("\nCovariance matrix (Age & Score):\n", df[["Age", "Score"]].cov())
"""
corr() measures linear relationship (Pearson by default).
cov() measures covariance between columns.
"""

"""_________________<SORTING & RANKING>_________________"""

import pandas as pd

# Sorting lets you order rows by column values.
# Ranking assigns relative positions based on values (like converting scores to positions).

# Sample DataFrame
df = pd.DataFrame(
    {
        "Name":  ["Alice", "Bob", "Cara", "Dan", "Eve"],
        "Age":   [25, 30, 22, 28, 35],
        "Score": [88.5, 92.0, 76.0, 85.5, 90.0],
    }
)
print("Original DataFrame:\n", df)
"""
Original DataFrame:
   Name  Age  Score
0  Alice   25   88.5
1    Bob   30   92.0
2   Cara   22   76.0
3    Dan   28   85.5
4    Eve   35   90.0
"""

# 1. Sorting rows with sort_values()

# a) Sort by a single column (ascending)
df_age_asc = df.sort_values(by="Age")
print("\nSorted by Age (ascending):\n", df_age_asc)
"""
sort_values(by='Age') sorts rows by Age from smallest to largest.
"""

# b) Sort by a single column (descending)
df_score_desc = df.sort_values(by="Score", ascending=False)
print("\nSorted by Score (descending):\n", df_score_desc)
"""
ascending=False sorts in descending order.
"""

# c) Sort by multiple columns (e.g., Score desc, then Age asc)
df_multi_sort = df.sort_values(by=["Score", "Age"], ascending=[False, True])
print("\nSorted by Score (desc) then Age (asc):\n", df_multi_sort)
"""
When values tie on Score, Age is used as tie-breaker.
"""

# 2. Sorting by index with sort_index()

df_index_sorted = df.set_index("Name").sort_index()
print("\nAfter setting Name as index and sorting by index:\n", df_index_sorted)
"""
sort_index() orders rows by index labels instead of column values.
"""

# 3. Ranking values with rank()

# a) Rank scores (highest score gets highest rank or lowest rank depending on 'ascending')

df["Score_rank_desc"] = df["Score"].rank(ascending=False, method="dense")
print("\nScore ranking (descending, 'dense' method):\n", df[["Name", "Score", "Score_rank_desc"]])
"""
rank(ascending=False) gives rank 1 to the highest value.
method='dense' ensures ranks increase by 1 without gaps when there are ties.
"""

# b) Different ranking method example
df["Age_rank_min"] = df["Age"].rank(ascending=True, method="min")
print("\nAge ranking (ascending, 'min' method):\n", df[["Name", "Age", "Age_rank_min"]])
"""
Common rank methods:
- 'average' (default): average rank for ties
- 'min'    : lowest rank in the group for all ties
- 'max'    : highest rank in the group for all ties
- 'dense'  : like 'min', but no gaps in ranks
"""

# 4. Sorting and ranking combined
df_sorted_ranked = df.sort_values(by="Score", ascending=False)
df_sorted_ranked["Score_rank"] = df_sorted_ranked["Score"].rank(ascending=False)
print("\nSorted by Score with added Score_rank:\n", df_sorted_ranked)
"""
Combining sorting and ranking helps in ordered analyses.
"""

"""_________________<GROUPBY & AGGREGATION>_________________"""

import pandas as pd

# groupby + aggregation let you compute summaries per group, similar to SQL GROUP BY.
# Common patterns:
# - df.groupby("col").agg(...)
# - Multiple keys, multiple aggregations

# Sample DataFrame
df = pd.DataFrame(
    {
        "City":  ["Delhi", "Delhi", "Mumbai", "Pune", "Delhi", "Mumbai"],
        "Segment": ["A", "B", "A", "C", "B", "A"],
        "Sales":  [100, 150, 200, 120, 180, 220],
        "Qty":    [1,   2,   3,   1,   2,   4],
    }
)
print("Original DataFrame:\n", df)
"""
Original DataFrame:
     City Segment  Sales  Qty
0   Delhi       A    100    1
1   Delhi       B    150    2
2  Mumbai       A    200    3
3    Pune       C    120    1
4   Delhi       B    180    2
5  Mumbai       A    220    4
"""

# 1. Simple groupby on one column

city_sum = df.groupby("City")["Sales"].sum()
print("\nTotal Sales per City:\n", city_sum)
"""
groupby('City')['Sales'].sum() gives a Series indexed by City with summed Sales.
"""

city_mean_qty = df.groupby("City")["Qty"].mean()
print("\nAverage Qty per City:\n", city_mean_qty)
"""
You can aggregate different columns separately using groupby(...)[col].
"""

# 2. Groupby with multiple keys

city_segment_sum = df.groupby(["City", "Segment"])["Sales"].sum()
print("\nTotal Sales per City & Segment:\n", city_segment_sum)
"""
Multi-key groupby returns a Series with a MultiIndex (City, Segment).
"""

# 3. Using .agg() for multiple aggregations

# a) Multiple aggregations on a single column
sales_agg = df.groupby("City")["Sales"].agg(["sum", "mean", "max"])
print("\nMultiple aggregations on Sales per City:\n", sales_agg)
"""
agg(['sum', 'mean', 'max']) computes several aggregations at once.
"""

# b) Different aggregations per column
multi_agg = df.groupby("City").agg(
    {
        "Sales": ["sum", "mean"],
        "Qty":   ["sum", "max"],
    }
)
print("\nDifferent aggregations per column:\n", multi_agg)
"""
Pass a dict to agg: {column_name: [list, of, functions]}.
"""

# 4. Resetting index after groupby

city_sales = df.groupby("City", as_index=False)["Sales"].sum()
print("\nGroupby with as_index=False (returns regular DataFrame):\n", city_sales)
"""
as_index=False returns a DataFrame with 'City' as a regular column instead of index.
"""

# 5. Using custom aggregation functions

def sales_range(x):
    return x.max() - x.min()

custom_agg = df.groupby("City")["Sales"].agg(
    total="sum",
    avg="mean",
    range=sales_range,
)
print("\nCustom-named aggregations on Sales per City:\n", custom_agg)
"""
You can assign names to aggregations by using keyword arguments in agg().
"""

# 6. Filtering groups after groupby

def filter_high_sales(group):
    return group["Sales"].sum() > 300
high_sales_cities = df.groupby("City").filter(filter_high_sales)
print("\nCities with total Sales > 300:\n", high_sales_cities)
"""
filter(...) keeps groups where the function returns True.
Useful for subsetting based on group-level criteria.
"""

# 7. Iterating over groups

for city, group in df.groupby("City"):
    print(f"\nCity: {city}\n", group)
"""
You can iterate over groups using a for loop.
Each iteration gives the group key and the corresponding DataFrame slice.
"""

# 8. Transforming data within groups
df["Sales_zscore"] = df.groupby("City")["Sales"].transform(
    lambda x: (x - x.mean()) / x.std()
)
print("\nSales z-score within each City:\n", df[["City", "Sales", "Sales_zscore"]])
"""
transform(...) returns a Series aligned with the original DataFrame, applying the function within each group.
Useful for standardizing or normalizing values per group.
"""

"""_________________<PIVOT TABLES & RESHAPING (MELT, STACK, UNSTACK)>_________________"""

import pandas as pd

# Pivoting and reshaping let you switch between:
# - "Long" / tidy format  (many rows, few columns)
# - "Wide" format         (fewer rows, more columns)
# Tools:
# - pivot, pivot_table
# - melt (unpivot)
# - stack, unstack

# Sample DataFrame in "long" format
df = pd.DataFrame(
    {
        "City":   ["Delhi", "Delhi", "Mumbai", "Mumbai", "Pune", "Pune"],
        "Year":   [2023,    2024,    2023,     2024,     2023,   2024],
        "Sales":  [100,     120,     200,      230,      90,    110],
    }
)
print("Original 'long' DataFrame:\n", df)
"""
Original 'long' DataFrame:
     City  Year  Sales
0   Delhi  2023    100
1   Delhi  2024    120
2  Mumbai  2023    200
3  Mumbai  2024    230
4    Pune  2023     90
5    Pune  2024    110
"""

# 1. pivot(): long -> wide (single value column, no duplicates in index/columns)

df_pivot = df.pivot(index="City", columns="Year", values="Sales")
print("\nPivot: rows=City, columns=Year, values=Sales:\n", df_pivot)
"""
pivot() reshapes so that:
- 'City' becomes the row index
- 'Year' becomes the column labels
- 'Sales' fills the table
"""

# 2. pivot_table(): similar to pivot, but allows duplicates + aggregation

# Example with duplicates to show aggregation
df_dup = pd.DataFrame(
    {
        "City":  ["Delhi", "Delhi", "Delhi", "Mumbai"],
        "Year":  [2023,    2023,    2024,    2023],
        "Sales": [100,     150,     120,     200],
    }
)
print("\nDataFrame with duplicate City-Year pairs:\n", df_dup)
"""
pivot() would fail here due to duplicate (City, Year) combinations.
pivot_table() solves this by aggregating.
"""

df_pivot_table = df_dup.pivot_table(
    index="City",
    columns="Year",
    values="Sales",
    aggfunc="mean",   # or 'sum', 'max', custom func, etc.
)
print("\nPivot table with mean Sales per City-Year:\n", df_pivot_table)
"""
pivot_table() aggregates duplicates using aggfunc.
"""

# 3. melt(): wide -> long (unpivot)

# Start from a wide format DataFrame
df_wide = df_pivot.reset_index()
print("\nWide-format DataFrame (City as column, Years as separate columns):\n", df_wide)
"""
   City  2023  2024
0  Delhi   100   120
1 Mumbai   200   230
2   Pune    90   110
"""

df_long_again = pd.melt(
    df_wide,
    id_vars="City",
    var_name="Year",
    value_name="Sales",
)
print("\nAfter melt (back to long format):\n", df_long_again)
"""
melt() takes columns (2023, 2024) and turns them into rows with 'Year' and 'Sales'.
"""

# 4. stack() and unstack() with MultiIndex

# Create a MultiIndex pivot table for demonstration
df_multi = df_dup.pivot_table(
    index="City",
    columns="Year",
    values="Sales",
    aggfunc="sum",
)
print("\nMultiIndex-like pivot table:\n", df_multi)
"""
Columns are a simple Index here (just Year), but stack/unstack treat them similarly.
"""

# a) stack(): move columns to a row-level index
stacked = df_multi.stack()   # default: stacks the last level of columns
print("\nAfter stack() (columns -> inner row index):\n", stacked)
"""
stack() converts columns into a new inner index level, producing a Series with MultiIndex.
"""

# b) unstack(): move an index level back to columns
unstacked = stacked.unstack()   # revert the previous stack
print("\nAfter unstack() (index level -> columns):\n", unstacked)
"""
unstack() is the inverse of stack for a given index level.
"""

# 5. Example: multi-dimension pivot_table with multiple agg functions

df_sales = pd.DataFrame(
    {
        "Region": ["North", "North", "South", "South", "North"],
        "City":   ["Delhi", "Delhi", "Mumbai", "Pune", "Delhi"],
        "Year":   [2023,    2024,    2023,     2024,    2023],
        "Sales":  [100,     120,     200,      90,     150],
        "Qty":    [1,       2,       3,        1,       2],
    }
)
print("\nSales DataFrame:\n", df_sales)

pt = df_sales.pivot_table(
    index=["Region", "City"],
    columns="Year",
    values="Sales",
    aggfunc=["sum", "mean"],
)
print("\nPivot table with Region, City as index and Year as columns (sum & mean Sales):\n", pt)
"""
This pivot_table:
- Uses multiple index levels: Region, City
- Uses multiple aggregations: sum and mean
- Spreads results across a MultiIndex columns axis
"""

# 6. Resetting index after pivot/unstack

df_reset = df_pivot.reset_index()
print("\nPivot table with reset index:\n", df_reset)
"""
reset_index() converts index levels back to regular columns.
"""

"""_________________<WINDOW FUNCTIONS & TIME-BASED ROLLING>_________________"""

import pandas as pd
import numpy as np

# Window functions operate over a "window" of rows:
# - rolling(...)   : fixed-size moving window (e.g., last 3 rows)
# - expanding(...) : from the start up to current row
# - ewm(...)       : exponentially weighted windows
# With time-based indices, rolling windows can use time offsets (e.g., '7D').

# 1. Simple rolling window on numeric index

s = pd.Series([10, 20, 30, 40, 50], name="value")
print("Original Series:\n", s)
"""
Original Series:
0    10
1    20
2    30
3    40
4    50
Name: value, dtype: int64
"""

# Rolling mean with window size 3
rolling_mean_3 = s.rolling(window=3).mean()
print("\nRolling mean with window=3:\n", rolling_mean_3)
"""
First two positions are NaN (not enough data points).
From index 2 onward, each value is mean of current and previous 2.
"""

# 2. Rolling operations on a time-indexed DataFrame

date_index = pd.date_range("2024-01-01", periods=7, freq="D")
df = pd.DataFrame(
    {
        "value": [10, 12, 9, 13, 15, 14, 16],
    },
    index=date_index,
)
print("\nTime series DataFrame:\n", df)
"""
Time series with daily frequency:
            value
2024-01-01     10
...
2024-01-07     16
"""

# a) Rolling mean over last 3 rows
df["rolling_mean_3"] = df["value"].rolling(window=3).mean()
print("\nRolling mean over last 3 rows:\n", df)
"""
window=3 uses row count (3 consecutive rows) regardless of time gaps.
"""

# b) Time-based rolling window: last 3 days
df["rolling_mean_3D"] = df["value"].rolling("3D").mean()
print("\nTime-based rolling mean over last 3 days ('3D'):\n", df)
"""
rolling('3D') uses a 3-day time window, based on the DatetimeIndex.
"""

# 3. Expanding windows (from start to current row)

df["expanding_mean"] = df["value"].expanding().mean()
print("\nExpanding mean from start up to current row:\n", df[["value", "expanding_mean"]])
"""
expanding() accumulates all rows from the beginning up to each position.
Good for cumulative averages or other incremental stats.
"""

# 4. Exponentially weighted windows (ewm)

df["ewm_mean_alpha_0.5"] = df["value"].ewm(alpha=0.5, adjust=False).mean()
print("\nExponentially weighted mean (alpha=0.5):\n", df[["value", "ewm_mean_alpha_0.5"]])
"""
ewm(alpha=...) gives more weight to recent observations.
alpha closer to 1 -> more weight on latest values.
"""

# 5. Rolling with multiple aggregations and min periods

# Use min_periods to require a minimum number of observations in the window
roll = df["value"].rolling(window=3, min_periods=2)
df["roll_mean_min2"] = roll.mean()
df["roll_std_min2"] = roll.std()
print("\nRolling mean/std with window=3, min_periods=2:\n", df[["value", "roll_mean_min2", "roll_std_min2"]])
"""
min_periods=2 means:
- At least 2 non-NaN values are required to compute a result.
- Otherwise, result is NaN for that position.
"""

# 6. Rolling on multiple columns in a DataFrame

df_multi = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
    }
)
df_multi_rolling = df_multi.rolling(window=2).mean()
print("\nRolling mean on multiple columns:\n", df_multi_rolling)
"""
Rolling mean is computed separately for each column.
"""

"""
When to use each:
- Use Series.map() when:
    * You have a simple mapping/dictionary to apply to each element in a single column.
    * Example: df["Name"].map({"Alice": "A", "Bob": "B"})
- Use DataFrame.apply(axis=0/1) when:
    * You need column-wise or row-wise computations combining multiple values.
    * Example: df.apply(lambda col: col.max() - col.min(), axis=0)
- Use DataFrame.applymap() when:
    * You truly need an element-wise transformation over many cells.
    * Example: df.applymap(lambda x: str(x).upper()) for text formatting.
Choose based on whether your function operates on single values, entire rows/columns, or the whole DataFrame.
"""