def configurations():
    
    parameters = {'Unsupervised': {
                                   'Isolation_Forest': {'contamination': 0.002, 'n_estimators': 200, 'max_samples': 0.7},
                                   'LocalOutlierFactor': {'contamination': 0.002, 'n_neighbors': 20, 'algorithm': 'auto', 
                                                          'leaf_size': 30, 'metric': 'minkowski', 'novelty': True },
                                   'One_Class_SVM': {'nu': 0.05, 'kernel': 'rbf', 'gamma': 'scale'},
                                   'EllipticEnvelope': {'contamination': 0.002},
                                },
                  'Clustering': {
                                   'KMeans': {'n_clusters': 5, 'random_state': 0},
                                   'Birch': {'threshold': 0.5, 'branching_factor': 50},
                                   'MiniBatchKMeans': {'n_clusters': 5, 'random_state': 0},
                                   'DBSCAN': {'eps': 0.5, 'min_samples': 5, 'metric': 'euclidean'},
                                   'GMM': {'n_components': 5, 'covariance_type': 'full'},
                               },
                  'DimensionalityReduction': {
                                   'PCA': {'n_components': 4},
                                   'TruncatedSVD': {'n_components': 4},
                                   'UMAP': {'n_components': 4},
                                   'FastICA': {'n_components': 4},
                                   'LocallyLinearEmbedding': {'n_components': 4, 'eigen_solver': 'dense'},
                                   },
                  'FeatureSelection': {
                                   'CatBoost': {'iterations': 500, 'depth': 8, 'learning_rate': 0.1,
                                                'save_snapshot': False, 'verbose': False},
                                   'XGBoost': {'n_estimators': 400, 'nthread': 24,'learning_rate': 0.01, 
                                               'max_depth': 3},
                                  } 
                   }
    
    return parameters
