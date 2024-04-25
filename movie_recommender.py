import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

"""
these two functions are my implementation of finding eigenvalues and
eigenvectores. however for large matrices they take too long to process.
therefore I have used numpy.linalg.eigh() instead to find the eigenvalues and eigenvectors.
"""

def eigen(matrix, num_iterations=100):
    n = len(matrix)
    eigenvalues = np.zeros(n, dtype=complex)
    eigenvectors = np.eye(n, dtype=complex)
    for _ in range(num_iterations):
        q, r = qr(matrix)
        matrix = np.dot(r, q)
    for i in range(n):
        eigenvalues[i] = matrix[i, i]
    return eigenvalues, eigenvectors


def qr(matrix):
    n = len(matrix)
    q = np.zeros_like(matrix)
    r = np.zeros_like(matrix)

    for k in range(n):
        v = matrix[k:, k]
        norm_v = np.sqrt(np.sum(np.abs(v) ** 2))
        if norm_v == 0:
            q[k:, k] = 0
        else:
            q[k:, k] = v / norm_v
        r[k, k:] = norm_v * np.conj(q[k, k:])
        matrix[k:, k:] -= np.outer(q[k:, k], np.conj(r[k, k:]))
    return q, r


def calculate_u(M):
    B = np.dot(M, M.T)

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    ncols = np.argsort(eigenvalues)[::-1]

    return eigenvectors[:, ncols]


def calculate_v_transpose(M):
    B = np.dot(M.T, M)

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    ncols = np.argsort(eigenvalues)[::-1]

    return eigenvectors[:, ncols].T


def calculate_sigma(M):
    if np.size(np.dot(M, M.T)) > np.size(np.dot(M.T, M)):
        new_M = np.dot(M.T, M)
    else:
        new_M = np.dot(M, M.T)

    eigenvalues, eigenvectors = np.linalg.eigh(new_M)
    eigenvalues = np.sqrt(eigenvalues)
    return eigenvalues[::-1]


if __name__ == '__main__':
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')

    user_id = int(input('Enter userId: '))

    user_movie = np.zeros((611, 9742))
    for i, index in enumerate(movies['movieId']):
        user_movie[0][i] = movies['movieId'][i]

    user_movie = pd.DataFrame(user_movie)
    copy = user_movie.copy()
    user_movie.columns = user_movie.iloc[0]

    for row in ratings.iterrows():
        user_movie.loc[int(row[1]['userId'])][row[1]['movieId']] = row[1]['rating']
    user_movie.columns = copy.columns

    seen_movies = user_movie.columns[user_movie.iloc[user_id].to_numpy().nonzero()[0]]

    user_movie.drop(index=0, inplace=True)
    np_user_movie = user_movie.to_numpy()

    # V = calculate_v_transpose(np_user_movie)
    # U = calculate_u(np_user_movie)
    # sigma = calculate_sigma(np_user_movie)

    """
    these lines are used to save the heavy calculated U S V matrices for later use
    
    # np.save("V.npy", V)
    # np.save("U.npy", U)
    # np.save("sigma.npy", sigma)
    """
    V = np.load("V.npy")
    U = np.load("U.npy")
    sigma = np.load("sigma.npy")
   


    u_k = U[:, :3]
    sigma = np.diag(sigma)
    sigma_k = sigma[:3, :3]
    V_k = V[:3,:]

    predicted_ratings = np.dot(np.dot(u_k, sigma_k), V_k)

    similarities = cosine_similarity(predicted_ratings[user_id - 1, np.newaxis], predicted_ratings)

    sorted_similarities = np.argsort(similarities[0])[::-1]

    movies_to_recommend = [movie_idx for movie_idx in range(len(predicted_ratings[user_id -1])) if
                           movie_idx not in seen_movies]

    recommendation_list = [movie_idx for movie_idx in sorted_similarities if movie_idx in movies_to_recommend]

    num_recommendations = 5

    recommendation_list = recommendation_list[:num_recommendations]

    print(f'Recommendation List for User {user_id}:\n')
    for row in recommendation_list:
        title = movies.iloc[row, 1]
        print(title)


