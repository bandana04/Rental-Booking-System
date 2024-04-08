import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pickle

class RentSystemModel:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.vectorizer = TfidfVectorizer()
        self.df = None
        self.tfidf_matrix_description = None
        self.load_data()

    def load_data(self):
        db = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        mycursor = db.cursor()
        mycursor.execute("SELECT description FROM rent")
        rows = mycursor.fetchall()
        data = {'description': [row[0] for row in rows]}
        self.df = pd.DataFrame(data)
        mycursor.close()
        db.close()
        self.tfidf_matrix_description = self.vectorizer.fit_transform(self.df['description'])

    def process_user_input(self, user_input):
        preprocessed_input = user_input.lower()
        user_vector = self.vectorizer.transform([preprocessed_input])
        return user_vector

    def recommend(self, user_input, n=5):
        user_pref_vector = self.process_user_input(user_input)
        similarity_scores = cosine_similarity(user_pref_vector, self.tfidf_matrix_description)
        top_n_indices = similarity_scores.argsort()[0][::-1][:n]
        top_n_results = self.df.iloc[top_n_indices]

        # Filter results to include only descriptions containing the user input keyword
        filtered_results = top_n_results[top_n_results['description'].str.contains(user_input.lower(), case=False)]

        return filtered_results['description'].tolist()



    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Example usage:
if __name__ == "__main__":
    # Initialize and train the model
    model = RentSystemModel(host="localhost", user="root", password="", database="rentsystem")

    # Save the trained model
    model.save_model("rent_model.pkl")

    # Load the trained model
    loaded_model = RentSystemModel.load_model("rent_model.pkl")

    # Get user input
    user_input = input("Enter your description: ")

    # Get recommendation
    recommendation = loaded_model.recommend(user_input)

    # Print recommendation
    print("Recommendation:", recommendation)
