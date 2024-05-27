import numpy as np
import pandas as pd
from atlantic.processing.analysis import Analysis
from featransform.processing.processor import AutoLabelEncoder, AutoIterativeImputer
from featransform.feature_engineering.anomalies import Anomaly_Engineering
from featransform.feature_engineering.clustering import Clustering_Engineering
from featransform.feature_engineering.dimensionality import PCAensemble
from featransform.optimizer.evaluator import Evaluation
from featransform.optimizer.selector import Selector
from featransform.configs.parameters import configurations

sec_conf = configurations()

class Featransform:
    def __init__(self,
                 configs : dict = sec_conf,
                 optimize_iters : int = 6, 
                 validation_split : float = 0.15):
        """
        Initialize the Featransform object.

        :param validation_split: The ratio to split the dataset into training and testing sets.
        :param optimize_iters: The number of iterations to optimize feature selection.
        :param configs: Configuration dictionary.
        """
        self.validation_split = validation_split
        self.optimize_iters = optimize_iters
        self.configs = configs
        self.iter_imputer = None # AutoIterativeImputer object for handling missing values through imputation
        self.encoder = None # AutoLabelEncoder object for encoding categorical variables
        self.selected_features, self.orig_cols, self.eng_cols = [], [], [] # Lists to store selected, original, and engineered features
        self.feature_importance = None # Variable to store feature importance information
        self.performance_history = None  # Variable to store performance history during feature selection optimization
        self.target = None # Target column variable
        self.dp = None # Analysis object for dataset analysis and processing
        self.cl = None # Clustering_Engineering object for clustering-based feature engineering
        self.ae = None # Anomaly_Engineering object for anomaly detection-based feature engineering
        self.pcae = None # PCAensemble object for dimensionality reduction using PCA models
        
        assert 0.05 <= validation_split <= 0.45, 'validation_split should be in [0.05, 0.45] interval'
        assert 0 <= optimize_iters <= 10 , 'optimize_iters value should be in [0, 10] interval'
        
        
    def fit_engineering(self,
                        X : pd.DataFrame = None,
                        target : str = None):
        """
        Fit various feature engineering steps.

        :param X: Input DataFrame.
        :param target: Target column.
        :return: Returns the class instance.
        """
        
        assert X.shape[0] >= 5 , 'Input columns should be at least of size 4'
        
        # Initialize Analysis, target, and create a copy of the input DataFrame
        self.dp, self.target, X_ = Analysis(target), target, X.copy()
        
        # Convert datetime columns to a standard format
        datetime_columns = X_.select_dtypes(include=[np.datetime64]).columns
        for col in datetime_columns:
            X_[col]=pd.to_datetime(X_[col].dt.strftime('%Y-%m-%d %H:%M:%S'))

        # Engineering date features and dropping Id type and constant columns
        X_ = self.dp.engin_date(X_, drop = True)
        X_ = X_.drop(columns=[col for col in X_.columns if X_[col].nunique() == len(X_) or X_[col].nunique() == 1])

        # Split the dataset into training and testing sets
        train,test = self.dp.split_dataset(X=X_,split_ratio=1-self.validation_split)
        train,test = train.reset_index(drop = True), test.reset_index(drop = True)
        train = train[[col for col in train.columns if col != self.target] + [self.target]] # target to last index position

         # Store original columns and categorical columns
        self.orig_cols, cat_cols = list(train.columns), list(self.dp.cat_cols(X = train))
        
        # Imputation step using AutoIterativeImputer
        if (train.isnull().sum().sum() or test.isnull().sum().sum()) != 0:
      
            ## Create Iterative Imputer
            self.iter_imputer = AutoIterativeImputer(initial_strategy = 'mean',
                                                     imputation_order = 'ascending')
            # Fit
            self.iter_imputer.fit(train)
            # Transform
            train = self.iter_imputer.transform(train.copy())
            test = self.iter_imputer.transform(test.copy())
        
        # Encoding categorical columns using AutoLabelEncoder
        if len(cat_cols)>0:

            ## Create Label Encoder
            self.encoder = AutoLabelEncoder()
            # Fit
            self.encoder.fit(train[cat_cols])
            # Transform
            train = self.encoder.transform(X = train)
            test = self.encoder.transform(X = test)

        # Split the data into training and testing sets after imputation and encoding
        X_train, X_test, y_train, y_test = self.dp.divide_dfs(train = train, test = test)
        
        ############################################### Feature Engineering
        # Anomaly Detection
        print('Fitting Anomaly Detection Ensemble')
        # Instantiate the Anomaly_Engineering class with unsupervised anomaly detection models
        self.ae = Anomaly_Engineering(det_models = list(self.configs['Unsupervised'].keys()),
                                      configs = self.configs,
                                      del_score = True)
        # Fit the unsupervised anomaly detection models on the training data
        self.ae.unsupervised_fit(X = X_train)
        # Predict anomalies for the training and testing datasets
        anomalies_train = self.ae.unsupervised_prediction(X = X_train)
        anomalies_test = self.ae.unsupervised_prediction(X = X_test)

        # Clustering
        print('Fitting Clustering Ensemble')
        # Instantiate the Clustering_Engineering class with clustering models
        self.cl = Clustering_Engineering(cluster_models = [c for c in list(self.configs['Clustering'].keys()) if c != 'DBSCAN'], 
                                         configs = self.configs)
        # Fit the clustering models on the training data
        self.cl.clustering_fit(X = X_train)
        # Predict clusters for the training and testing datasets
        clustering_train = self.cl.clustering_prediction(X = X_train)
        clustering_test = self.cl.clustering_prediction(X = X_test)

        # PCAs Ensemble
        print("Fitting PCA's Ensemble")
        # Instantiate the PCAensemble class with PCA models
        self.pcae = PCAensemble(pca_models = list(self.configs['DimensionalityReduction'].keys()), 
                                configs = self.configs)
        # Fit the PCA models on the training data
        self.pcae.dimensionality_fit(X = X_train)
        # Transform the training and testing datasets using PCA
        pcas_train = self.pcae.dimensionality_transform(X = X_train)
        pcas_test = self.pcae.dimensionality_transform(X = X_test)
        
        #####################################
        # Feature Engineered Datasets
        # Concatenate the anomalies, clustering, and PCA transformed features with the original datasets
        train_concat = pd.concat([pd.concat([anomalies_train, clustering_train, pcas_train], axis=1),train],axis=1) 
        test_concat = pd.concat([pd.concat([anomalies_test, clustering_test, pcas_test], axis=1),test],axis=1)  

        #####################################
        # Evaluation
        # If optimization iterations are less than or equal to 1, use all features
        if self.optimize_iters <= 1:
            self.selected_features = list(train_concat.columns)
        else:
            print(' ')
            print('Feature Selection Optimization: ')
            # Instantiate the Evaluation class for feature selection optimization
            ev = Evaluation(train = train_concat,
                            test = test_concat,
                            target = target,
                            optimize_iters = self.optimize_iters,
                            configs = self.configs)
            # Perform feature selection optimization and get the selected features
            self.selected_features = ev.feature_upgrading()
            # Save performance history and feature importance from the evaluation
            self.performance_history, self.feature_importance = ev.performance_history, ev.va_imp
        # Identify the engineered features
        self.eng_cols = [col for col in self.selected_features if col not in self.orig_cols]
        
        return self
        
    def transform(self, X : pd.DataFrame):
        """
        Transform the input DataFrame using the fitted feature engineering steps.

        :param X: Input DataFrame.
        :return: Transformed DataFrame.
        """
        # Create a copy of the input DataFrame
        X_ = X.copy()

        # Reset index for consistency
        X_ = X_.reset_index(drop = True)

        # Convert datetime columns to a standard format
        datetime_columns = X_.select_dtypes(include=[np.datetime64]).columns
        for col in datetime_columns:
            X_[col]=pd.to_datetime(X_[col].dt.strftime('%Y-%m-%d %H:%M:%S'))

        # Engineering date features
        X_ = self.dp.engin_date(X_, drop = True)

        # Extract original columns for the transformed DataFrame
        X_ = X_[self.orig_cols]
        X_orig = X_.copy()

        # Imputation step using AutoIterativeImputer if fitted during training
        if self.iter_imputer is not None:
            X_ = self.iter_imputer.transform(X = X_.copy())

        # Encoding categorical columns using AutoLabelEncoder if fitted during training
        if self.encoder is not None:
            X_ = self.encoder.transform(X = X_.copy())

        # Extract features for anomaly detection, clustering, and PCA transformation
        X_input= X_[[col for col in X_.columns if col != self.target]]
        
        # Anomalies
        anomalies = self.ae.unsupervised_prediction(X = X_input)
        # Clustering
        clustering = self.cl.clustering_prediction(X = X_input)
        # PCA's
        pcas = self.pcae.dimensionality_transform(X = X_input)
        
        # Concatenate the transformed features with the original DataFrame
        X_ = pd.concat([pd.concat([anomalies, clustering, pcas], axis=1),X_orig],axis=1)
        
        # Select only the engineered and selected features
        X_ = X_[self.selected_features]
        # Reorder columns to match the original order
        X_ = X_[list(set(self.orig_cols).intersection(self.selected_features)) + 
                        [col for col in X_.columns 
                                 if col not in list(set(self.orig_cols).intersection(self.selected_features))]]
        
        return X_
    

__all__ = [
    'AutoLabelEncoder',
    'AutoIterativeImputer',
    'Anomaly_Engineering',
    'Clustering_Engineering',
    'PCAensemble',
    'Evaluation',
    'Selector',
    'configurations'
]




