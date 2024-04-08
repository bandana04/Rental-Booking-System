import mysql.connector
import pandas as pd
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="rentsystem"
)
mycursor = db.cursor()
mycursor.execute("SELECT price FROM rent")
rows = mycursor.fetchall()
# Convert the query result to a DataFrame
data = {
    'price': [row[0] for row in rows]
}
df = pd.DataFrame(data)

# Close the cursor and database connection
mycursor.close()
db.close()
user_price = int(input("Enter your price: "))
def filter_by_price(df, user_price, threshold=5000):
    lower_bound = user_price - threshold
    upper_bound = user_price + threshold
    filtered_df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
    return filtered_df
filtered_df = filter_by_price(df, user_price)
print(filtered_df['price'].values[0])