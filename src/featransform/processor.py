import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
       
class AutoLabelEncoder(TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.columns = None
    
    def fit(self, X, y = None):
        self.columns = X.columns
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            #Store the original classes and add a new label for unseen values
            le.classes_ = np.append(le.classes_, 'Unknown')
            self.label_encoders[col] = le
            
        return self
    
    def transform(self, X, y = None):
        if self.columns is None:
            raise ValueError('The transformer has not been fitted yet.')
            
        X_encoded = X.copy()
        for col in self.columns:
            le = self.label_encoders[col]
            # Use the 'UNKNOWN' label for previously unseen values
            X_encoded[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(['Unknown'])[0])
            
        return X_encoded
    
    def inverse_transform(self, X):
        if self.columns is None:
            raise ValueError('The transformer has not been fitted yet.')
        
        X_decoded = X.copy()
        for col in self.columns:
            le = self.label_encoders[col]
            X_decoded[col] = le.inverse_transform(X = X[col])
            
        return X_decoded
    
class AutoIterativeImputer(TransformerMixin):

    def __init__(self, max_iter : int = 10, 
                 random_state : int = None, 
                 initial_strategy : str = 'mean', # {'mean', 'median', 'most_frequent', 'constant'}
                 imputation_order : str = 'ascending', # {'ascending', 'descending', 'roman', 'arabic', 'random'}
                 target : str = None):
        self.max_iter = max_iter
        self.random_state = random_state
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.target = target
        self.imputer = None
        self.numeric_columns = None  # Store the numeric columns here

    def fit(self, X, y = None):
        if self.target is not None:
            X = X.drop(columns = [self.target])
        
        # Detect numeric columns
        self.numeric_columns = X.select_dtypes(include = [np.number]).columns.tolist()
        
        # Fit the imputer on numeric columns only
        self.imputer = IterativeImputer(max_iter = self.max_iter,
                                        random_state = self.random_state,
                                        initial_strategy = self.initial_strategy,
                                        imputation_order = self.imputation_order)
        self.imputer.fit(X = X[self.numeric_columns])
        
        return self

    def transform(self, X):
        if self.imputer is None:
            raise ValueError("You must call 'fit' first to initialize the imputer.")
        
        # Transform only the numeric columns using the fitted imputer
        imputed_numeric = self.imputer.transform(X[self.numeric_columns])
        
        # Update the original DataFrame with imputed values
        X[self.numeric_columns] = imputed_numeric
        
        return X

        



