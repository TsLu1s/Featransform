from featransform.processor import AutoLabelEncoder, AutoIterativeImputer
from featransform.clustering import Clustering_Engineering
from featransform.anomalies import Anomaly_Engineering
from featransform.dimensionality import PCAensemble
from featransform.parameters import configurations
from featransform.selector import Selector
from atlantic.analysis import Analysis
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore', category=Warning)


######################################################  Load Data

# source_data = "https://www.kaggle.com/datasets/surekharamireddy/fraudulent-claim-on-cars-physical-damage"

url="https://raw.githubusercontent.com/TsLu1s/Atlantic/main/data/Fraudulent_Claim_Cars_class.csv"
data = pd.read_csv(url) # Dataframe Loading Example

target="fraud"
data[target]=data[target].astype('category')

######################################################

# Drop null values from target column

data.dropna(subset=[target], inplace=True)

# Drop constant columns directly and ID type columns directly

data = data.drop(columns=[col for col in data.columns if data[col].nunique() == len(data) or data[col].nunique() == 1])

# Split Data into Train and Test subsets

train,test = train_test_split(data, train_size = 0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True)

train.isna().sum()
train.dtypes

######################################################
####### AutoIterativeImputer - Null Imputation

if (train.isnull().sum().sum() or test.isnull().sum().sum()) != 0:
    
    ## Create Label Encoder
    iter_imputer = AutoIterativeImputer(initial_strategy='mean')
    ## Fit
    iter_imputer.fit(train)
    ## Transform
    train = iter_imputer.transform(train.copy())
    test = iter_imputer.transform(test.copy())

####### Encoders Application - Encoding Application

cat_cols=[col for col in train.select_dtypes(include=['object','category']).columns if col != target]

if len(cat_cols)>0:
    ## Create Label Encoder
    encoder = AutoLabelEncoder()
    ## Fit
    encoder.fit(train[cat_cols])
    ## Transform
    train = encoder.transform(X = train)
    test = encoder.transform(X = test)

######################################################
############ Customizable Feature Engineering 

dp = Analysis(target = target)

# Datetime Feature Engineering

train, test = dp.engin_date(train, drop=True), dp.engin_date(train, drop=True)

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = dp.divide_dfs(train=train, test=test)

# Loading Configs Parameters 

configs = configurations()
print(configs)

### Anomaly Detection Ensemble Implementation
# Anomaly Detection Customization Example

configs['Unsupervised']['Isolation_Forest']['n_estimators'] = 300

ae = Anomaly_Engineering(det_models = ['Isolation_Forest',
                                       'LocalOutlierFactor',
                                       'One_Class_SVM', 
                                       'EllipticEnvelope'],
                        configs = configs,
                        del_score = True)
ae.unsupervised_fit(X = X_train)
anomalies_train = ae.unsupervised_prediction(X = X_train)
anomalies_test = ae.unsupervised_prediction(X = X_test)

### Clustering Ensemble Implementation
# Clustering Customization Example

configs['Clustering']['KMeans']['n_clusters'] = 3

cl = Clustering_Engineering(cluster_models = ['KMeans',
                                              'Birch',
                                              'MiniBatchKMeans',
                                              'GMM'],
                            configs = configs)
cl.clustering_fit(X = X_train)
clustering_train = cl.clustering_prediction(X = X_train)
clustering_test = cl.clustering_prediction(X = X_test)

### Dimensionality Reduction Ensemble Implementation
# Dimensionality Reduction Customization Example

configs['DimensionalityReduction']['UMAP']['n_components'] = 6

pcae = PCAensemble(pca_models = ['PCA',
                                 'TruncatedSVD',
                                 'UMAP',
                                 'FastICA',
                                 'LocallyLinearEmbedding'],
                   configs = configs)
pcae.dimensionality_fit(X = X_train)
pcas_train = pcae.dimensionality_transform(X = X_train)
pcas_test = pcae.dimensionality_transform(X = X_test)

### Feature Engineering Concatenation

feat_train = pd.concat([clustering_train, anomalies_train, pcas_train], axis=1)
feat_test = pd.concat([clustering_test, anomalies_test, pcas_test], axis=1)

### Concatenation with Original Datasets

train_concat = pd.concat([train,feat_train],axis=1)
test_concat = pd.concat([test,feat_test],axis=1)

############ Feature Selection - Evaluate Feature Importance

configs['FeatureSelection']['CatBoost']['iterations'] = 1000
configs['FeatureSelection']['XGBoost']['n_estimators'] = 500

selector = Selector(configs = configs)

va_imp = selector.feature_selection(X = train_concat, 
                                    target = target)
