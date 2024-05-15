#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import numpy as np
from scipy import spatial
from scipy import stats

class KNN:
    """
    Implementation of the k-nearest neighbors algorithm for classification
    and regression problems.
    """
    def __init__(self, k, aggregation_function):
        """
        Initializes a k-nearest neighbors classifier/regressor.

        Parameters
        ----------
        k : int
            Number of nearest neighbors to consider for prediction.

        aggregation_function : str
            The aggregation function to use for prediction.
            - "mode" for classification
            - "average" for regression
        """
        self.k = k
        self.aggregation_function = aggregation_function
        if aggregation_function not in ["mode", "average"]:
            raise ValueError("Invalid aggregation function. Choose 'mode' for classification or 'average' for regression.")
        

    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).

        Parameters
        ----------
        X : 2D-array of shape (n_samples, n_features)
            Training/Reference data.
        y : 1D-array of shape (n_samples,)
            Target values.
        """
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        """
        Predicts the output variable's values for the query points X.

        Parameters
        ----------
        X : 2D-array of shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        y : 1D-array of shape (n_queries,)
            Class labels for each query.
        """
        predictions = []
        for query_point in X:
            # Calculate distances between the query point and all training points
            distances = [np.linalg.norm(query_point - train_point) for train_point in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]

            # Get the corresponding target values for the k nearest neighbors
            nearest_targets = self.y_train[nearest_indices]

            if self.aggregation_function == 'mode':
                # Use mode for classification
                prediction = stats.mode(nearest_targets).mode
            elif self.aggregation_function == 'average':
                # Use average for regression
                prediction = np.mean(nearest_targets)

            predictions.append(prediction)
        return np.array(predictions)
    

