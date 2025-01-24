# Book Recommender System

Overview

The Book Recommender System suggests books to users based on the similarity between books. The system is implemented using two algorithms: TF-IDF (Term Frequency-Inverse Document Frequency) and K-Nearest Neighbors (KNN). It is an efficient tool for finding books that match a user's interests.

Features

Recommends books based on the title input.

Uses TF-IDF to calculate the textual similarity between book descriptions or metadata.

Leverages KNN to identify and rank similar books.

Interactive and user-friendly front-end built using HTML, CSS, and Flask.

Algorithms Used

1. TF-IDF

TF-IDF is used to represent book descriptions as feature vectors. It highlights important terms while reducing the weight of commonly used words.

2. KNN

KNN is employed to find books similar to the input title. It identifies the top k most similar books based on their feature vectors.


Run the application:

python app.py

Open your browser and visit http://127.0.0.1:5000 to use the recommender system.
