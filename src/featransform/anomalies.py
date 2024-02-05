import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from Featransform_dev.parameters import configurations

sec_conf = configurations()

class Anomaly_Engineering:
    def __init__(self,
                 det_models : list = ['Isolation_Forest'], # List of anomaly detection models
                 configs : dict = sec_conf, # Configuration dictionary for models
                 del_score : bool = False): # Flag to delete or integrate anomaly scores in predictions
        self.det_models = det_models
        self.configs = configs
        self.del_score = del_score
        self.uns_model = None
        self.unsupervised_dict = {} # Dictionary to store fitted unsupervised models

    def unsupervised_fit(self,
                         X : pd.DataFrame):
        """
        Fit unsupervised anomaly detection models to the input data.

        :param X: Input DataFrame for training models.
        :return: Returns the class instance.
        """

        for model in self.det_models:
            params = self.configs['Unsupervised'][model]
            
            if model == 'Isolation_Forest':
                
                self.uns_model = IsolationForest(**params)
                self.uns_model.fit(X = X)
                
            elif model == 'LocalOutlierFactor':
                
                self.uns_model = LocalOutlierFactor(**params)
                self.uns_model.fit(X = X)
                
            elif model == 'One_Class_SVM':
                
                self.uns_model = OneClassSVM(**params)
                self.uns_model.fit(X = X)
            
            elif model == 'EllipticEnvelope':
                
                self.uns_model = EllipticEnvelope(**params)
                self.uns_model.fit(X = X)

            # Store the fitted model information in a dictionary
            if len(self.unsupervised_dict.keys()) == 0:
                self.unsupervised_dict = {model : self.uns_model}
            elif len(self.unsupervised_dict.keys()) > 0:
                detector = {model : self.uns_model}
                self.unsupervised_dict.update(detector)
        
        return self
    
    def unsupervised_prediction(self,
                                X : pd.DataFrame,):
        """
        Fit unsupervised anomaly detection models to the input data.

        :param X: Input DataFrame for training models.
        :return: Returns the class instance.
        """                            
        results = pd.DataFrame()
        for model in self.unsupervised_dict.keys(): 
            self.uns_model = self.unsupervised_dict[model]
            output = pd.Series(self.uns_model.predict(X = X)).apply(lambda x: 1 if x == -1 else 0) # 0 good value, 1 is an anomaly
            score = self.uns_model.decision_function(X = X)
            
            if model=='Isolation_Forest':
                col_str='IF'
            if model=='LocalOutlierFactor':
                col_str='LOF'
            if model=='One_Class_SVM':
                col_str='SVM'
            if model=='EllipticEnvelope':
                col_str='ENV'

            results = pd.concat([results, pd.DataFrame({col_str + '_method': output, col_str + 'score': score})], axis=1)
        if self.del_score == True : results = results.filter(like='_method', axis=1)
        return results
        