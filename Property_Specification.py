import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

db = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "",
    database = "rentsystem"
)
mycursor = db.cursor()
print(db)
mycursor.execute("SELECT Property_Specification FROM rent")
rows = mycursor.fetchall()
# Convert the query result to a DataFrame
data = {
    'Property_Specification': [row[0] for row in rows],
}
df = pd.DataFrame(data)

# Close the cursor and database connection
mycursor.close()
db.close()


# Assuming df['Property_Specification'] contains numerical data

# Convert numerical data to categorical data
df['Property_Specification'] = df['Property_Specification'].astype(str)

# Initialize the encoder
encoder = OneHotEncoder()

# Fit and transform the 'Property_Specification' column
one_hot_encoded_Property_Specification = encoder.fit_transform(df[['Property_Specification']])
from sklearn.preprocessing import OneHotEncoder

# Initialize the encoder
encoder = OneHotEncoder()

# Fit and transform the 'Property_Specification' column
one_hot_encoded_Property_Specification = encoder.fit_transform(df[['Property_Specification']])

# Define a function to process user input and one-hot encode it
def process_user_input(user_input, encoder):
    # Convert user input to a 2D array with a single column
    user_input_2d = [[user_input]]

    # Encode the user input
    user_encoded = encoder.transform(user_input_2d)

    return user_encoded

# Example usage:
user_input = input("Enter your Property_Specification: ")  # Get user input
user_pref_encoded = process_user_input(user_input, encoder)


# Assuming user_pref_vector is the vectorized version of the user's preference
similarity_scores = cosine_similarity(user_pref_encoded, one_hot_encoded_Property_Specification)
# Get the index of the top result
top_1_index = similarity_scores.argsort()[0][-1]

# Select the top result from the DataFrame
top_1_result = df.iloc[[top_1_index]]

print(top_1_result['Property_Specification'].values[0])