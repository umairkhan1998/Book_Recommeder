import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
app = Flask(__name__)


data = pd.read_csv("book_updated.csv",encoding='latin1')

# Replace NaN values with an empty string or a placeholder
data['Title'] = data['Title'].fillna('')
data['Authors'] = data['Authors'].fillna('')
data['Description'] = data['Description'].fillna('')


# Function to combine book title, author, and description
def combine_features(row):
    return row['Title'] + ' ' + row['Authors'] + ' ' + row['Description']


# Apply the function to create the 'title_author_desc' feature
data['title_author_desc'] = data.apply(combine_features, axis=1)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['title_author_desc'])

# Initialize the NearestNeighbors model
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')

# Fit the model on the tfidf_matrix
nn_model.fit(tfidf_matrix)


def get_book_recommendations(book_title, n=8):
    # Combine the input book title, author, and description
    input_feature = book_title

    # Transform the input feature to the same vector space as the tfidf_matrix
    input_vector = tfidf_vectorizer.transform([input_feature])

    # Find the nearest neighbors for the input vector
    distances, indices = nn_model.kneighbors(input_vector, n_neighbors=n + 1)

    # Get the details of the top n most similar books
    similar_books = data.iloc[indices.flatten()[1:]]  # Exclude the first one as it is the input book itself

    return similar_books


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.json
    book_title = content['title']
    n = content.get('n', 8)  # Default to 8 recommendations if not specified
    recommendations = get_book_recommendations(book_title, n)
    result = recommendations[['Title','Authors']].to_dict(orient='records')
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)