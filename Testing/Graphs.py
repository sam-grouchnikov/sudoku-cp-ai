import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("performance_data.csv")

# Column names (assuming 7 columns)
# If your CSV has headers, these will match automatically.
x = df.iloc[:, 0]     # first column

# First figure: cols 2–4
plt.figure()
plt.plot(x, df.iloc[:, 1])
plt.plot(x, df.iloc[:, 2])
plt.plot(x, df.iloc[:, 3])
plt.xlabel("X")
plt.ylabel("Times")
plt.title("Times")
plt.legend(df.columns[1:4])
plt.show()

# Second figure: cols 5–7
plt.figure()
plt.plot(x, df.iloc[:, 4])
plt.plot(x, df.iloc[:, 5])
plt.plot(x, df.iloc[:, 6])
plt.xlabel("X")
plt.ylabel("Branches")
plt.title("Branches")
plt.legend(df.columns[4:7])
plt.show()
