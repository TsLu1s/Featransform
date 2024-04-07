import pandas as pd
from sklearn.cluster import (DBSCAN,
                             KMeans,
                             Birch,
                             MiniBatchKMeans)
from sklearn.mixture import GaussianMixture
from featransform.configs.parameters import configurations

sec_conf = configurations()

class Clustering_Engineering:
    def __init__(self, 
                 cluster_models: list = ['KMeans'],
                 configs : dict = sec_conf):
        """
        Initialize the Clustering_Engineering class.

        :param cluster_models: List of clustering models to be used.
        :param configs: Configuration dictionary for clustering models.
        """
        self.cluster_models = cluster_models
        self.configs = configs
        self.cluster_model = None # Placeholder for the clustering model
        self.clustering_dict = {} # Dictionary to store fitted clustering models

    def clustering_fit(self, X : pd.DataFrame):
        """
        Fit clustering models to the input data.

        :param X: Input DataFrame for training models.
        :return: Returns the class instance.
        """
        for model in self.cluster_models:
            params = self.configs['Clustering'][model]
            
            if model == 'KMeans':
                
                self.cluster_model = KMeans(**params)
                self.cluster_model.fit(X = X)
                
            elif model == 'Birch':

                self.cluster_model = Birch(**params)
                self.cluster_model.fit(X = X)
    
            elif model == 'MiniBatchKMeans':

                self.cluster_model = MiniBatchKMeans(**params)
                self.cluster_model.fit(X = X)
    
            elif model == 'DBSCAN':

                self.cluster_model = DBSCAN(**params)
                self.cluster_model.fit(X = X)
    
            elif model == 'GMM':

                self.cluster_model = GaussianMixture(**params)
                self.cluster_model.fit(X = X)

            # Store the fitted model information in a dictionary
            if len(self.clustering_dict.keys()) == 0:
                self.clustering_dict = {model : self.cluster_model}
            elif len(self.clustering_dict.keys())>0:
                cluster = {model : self.cluster_model}
                self.clustering_dict.update(cluster)

        return self

    def clustering_prediction(self, X : pd.DataFrame):
        """
        Generate clustering predictions using fitted models.

        :param X: Input DataFrame for making predictions.
        :return: DataFrame with clustering predictions.
        """       
        results = pd.DataFrame()

        # Iterate through fitted models
        for model in self.clustering_dict.keys(): 
            self.cluster_model = self.clustering_dict[model]
            
            # Generate predictions based on the model type
            if isinstance(self.cluster_model, (DBSCAN)):
                predictions = pd.Series(self.cluster_model.labels_).apply(lambda x: 1 if x == -1 else 0)
            else:
                predictions = self.cluster_model.predict(X = X)
            # Concatenate results to the output DataFrame
            results = pd.concat([results, pd.DataFrame({model: predictions})], axis=1)

        return results