import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

db = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "",
    database = "rentsystem"
)
mycursor = db.cursor()
print(db)
mycursor.execute("SELECT rent_type FROM rent")
rows = mycursor.fetchall()
# Convert the query result to a DataFrame
data = {
    'rent_type': [row[0] for row in rows],
}
df = pd.DataFrame(data)

# Close the cursor and database connection
mycursor.close()
db.close()


# Initialize the vectorizer
vectorizer = TfidfVectorizer()


# Fit and transform the 'location' column to get the TF-IDF matrix
tfidf_matrix_rent_type = vectorizer.fit_transform(df['rent_type'])
# print(tfidf_matrix_rent_type)
# Define a function to process user input and vectorize it
def process_user_location(user_input, vectorizer):
    # Preprocess user input (if needed)
    preprocessed_input = user_input  # For simplicity, converting to lowercase
    
    # Vectorize the preprocessed input
    user_vector = vectorizer.transform([preprocessed_input])
    
    return user_vector

# Example usage:
user_input = input("Enter the rent_type: ")
user_pref_vector = process_user_location(user_input, vectorizer)


# Assuming user_pref_vector is the vectorized version of the user's preference
similarity_scores = cosine_similarity(user_pref_vector, tfidf_matrix_rent_type)
# Get the index of the top result
top_1_index = similarity_scores.argsort()[0][-1]

# Select the top result from the DataFrame
top_1_result = df.iloc[[top_1_index]]

print(top_1_result['rent_type'].values[0])