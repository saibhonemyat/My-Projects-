"""
utilities.py
------------
Helper functions for the Movie Recommendation System.
Author: Sai Bhone Myat Naing
Edinburgh Napier University, 2022
"""

import numpy as np
import pandas as pd


def load_data(data_directory):
    """
    Load and return users, movies, and ratings dataframes.

    Parameters
    ----------
    data_directory : str
        Path to the folder containing the cleaned .dat files.

    Returns
    -------
    tuple : (usersdata, moviesdata, ratingsdata, modeltrainingratingsdata)
    """
    import os

    usersdata = pd.read_csv(
        os.path.join(data_directory, 'users-cleaned.dat'),
        sep='\t', header=None, encoding='latin-1'
    )
    moviesdata = pd.read_csv(
        os.path.join(data_directory, 'movies-cleaned.dat'),
        delimiter='\t', header=None, encoding='latin-1'
    )
    ratingsdata = pd.read_csv(
        os.path.join(data_directory, 'ratings-cleaned.dat'),
        delimiter='\t', header=None, encoding='latin-1'
    )
    modeltrainingratingsdata = np.array(
        pd.read_csv(os.path.join(data_directory, 'ratings-cleaned.dat'), sep='\t'),
        dtype='int'
    )

    moviesdata.columns = ['moviesID', 'Titles', 'Genre']

    return usersdata, moviesdata, ratingsdata, modeltrainingratingsdata


def dataconverting(data, numbers_of_users, numbers_of_movies):
    """
    Convert raw ratings array into a user-movie rating matrix.

    Each row is a user. Each column is a movie.
    Unrated movies are represented as 0.

    Parameters
    ----------
    data : np.ndarray
        Raw ratings data with columns [UserID, MovieID, Rating, ...].
    numbers_of_users : int
    numbers_of_movies : int

    Returns
    -------
    list of lists : shape (users, movies)
    """
    newdata = []
    for userid in range(1, numbers_of_users + 1):
        moviesid = data[:, 1][data[:, 0] == userid]
        ratingsid = data[:, 2][data[:, 0] == userid]
        ratings = np.zeros(numbers_of_movies)
        ratings[moviesid - 1] = ratingsid
        newdata.append(list(ratings))
    return newdata


def split_train_and_valid(data, ratio):
    """
    Split a user-movie rating matrix into training and validation sets.

    Uses Bernoulli sampling — each observed rating goes to training
    with probability `ratio`, otherwise to validation.

    Parameters
    ----------
    data : list of lists
        User-movie rating matrix.
    ratio : float
        Proportion of ratings to assign to training set (e.g. 0.6).

    Returns
    -------
    list : [training_set, validation_set]
    """
    training_set = np.zeros((len(data), len(data[0]))).tolist()
    validation_set = np.zeros((len(data), len(data[0]))).tolist()

    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] > 0:
                if np.random.binomial(1, ratio, 1):
                    training_set[i][j] = data[i][j]
                else:
                    validation_set[i][j] = data[i][j]

    return [training_set, validation_set]


def top_rank(titles, ratings, k=10):
    """
    Return the top-k titles sorted by descending predicted rating.

    Parameters
    ----------
    titles : np.ndarray
        Array of movie title strings.
    ratings : np.ndarray
        Array of predicted rating scores.
    k : int
        Number of top results to return.

    Returns
    -------
    tuple : (top_titles, top_ratings)
    """
    desc_ranks = np.argsort(ratings)[::-1]
    return titles[desc_ranks][:k], ratings[desc_ranks][:k]


def format_recommendations(titles, scores, user_id, topnumber):
    """
    Format recommendation results as a DataFrame.

    Parameters
    ----------
    titles : np.ndarray
    scores : np.ndarray
    user_id : int
    topnumber : int

    Returns
    -------
    pd.DataFrame
    """
    import numpy as np

    print(f"\nTop {topnumber} highest rating movies for User_ID [ {user_id} ]")
    df = pd.DataFrame(
        np.matrix((titles, scores)).T,
        (np.arange(topnumber) + 1).tolist(),
        columns=['Titles', 'Predicted Rating']
    )
    return df
