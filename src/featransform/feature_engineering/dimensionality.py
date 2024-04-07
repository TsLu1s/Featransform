import pandas as pd
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning, module = "tensorflow")
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import LocallyLinearEmbedding
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable TensorFlow optimizations using OneDNN
from umap import UMAP
from featransform.configs.parameters import configurations

# Load default configurations
sec_conf = configurations()

class PCAensemble:
    def __init__(self,
                 pca_models : list = ['PCA'], 
                 configs : dict = sec_conf):
        """
        Initialize the PCAensemble class.

        :param pca_models: List of PCA models to be used.
        :param configs: Configuration dictionary for PCA models.
        """
        self.pca_models = pca_models
        self.configs = configs
        self.dimensionality_dict = {} # Dictionary to store fitted PCA models
        self.pca_model = None # Placeholder for the last PCA model
        
    def dimensionality_fit(self,
                           X : pd.DataFrame):
        """
        Fit PCA models to the input data.

        :param X: Input DataFrame for training PCA models.
        :return: Returns the class instance.
        """
        for model in self.pca_models:
            params=self.configs['DimensionalityReduction'][model]
            
            if model == 'PCA': # PCA 1 || Linear Method : Eigendecomposition of the covariance matrix:

                self.pca_model = PCA(**params)
                self.pca_model.fit(X = X)
                
            elif model == 'TruncatedSVD': # PCA 2 || Linear Method : Singular value decomposition of the data matrix:

                self.pca_model = TruncatedSVD(**params)
                self.pca_model.fit(X = X)
                
            elif model == 'UMAP': # PCA 3 || Non-Linear Method : UMAP for dimensionality reduction

                self.pca_model = UMAP(**params)
                self.pca_model.fit(X = X)
            
            elif model == 'FastICA': # PCA 4 || Non-Linear Method : Independent Component Analysis (ICA):

                self.pca_model = FastICA(**params)
                self.pca_model.fit(X = X)
            
            elif model == 'LocallyLinearEmbedding': # PCA 5 || Non-Linear Method : Locally Linear Embedding (LLE)

                self.pca_model = LocallyLinearEmbedding(**params)
                self.pca_model.fit(X = X)
    
            # Store the fitted model information in a dictionary
            if len(self.dimensionality_dict.keys()) == 0:
                self.dimensionality_dict = {model : self.pca_model}
            elif len(self.dimensionality_dict.keys()) > 0:
                dimension = {model : self.pca_model}
                self.dimensionality_dict.update(dimension)
        
        return self
    
    def dimensionality_transform(self,
                                  X : pd.DataFrame):
        """
        Transform input data using fitted PCA models.

        :param X: Input DataFrame for transformation.
        :return: DataFrame with transformed features.
        """
        pca_list = []
        # Iterate through fitted models and transform input data
        for model, method in zip(self.dimensionality_dict.keys(), ['pca', 'svd', 'umap', 'ica', 'lla']): 
            self.pca_model = self.dimensionality_dict[model]
            input_data = self.pca_model.transform(X = X)

            # Create DataFrame with transformed features and appropriate column names
            pcas = pd.DataFrame(input_data, columns = [f'{method}_{i+1}' for i in range(input_data.shape[1])])
            pca_list.append(pcas)

        # Concatenate transformed features from different PCA models
        results = pd.concat(pca_list, axis=1)
        results = results.astype(float)
        
        return results
