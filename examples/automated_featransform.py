from Packages_Dev.Featransform_dev.processor import AutoLabelEncoder, AutoIterativeImputer
from Featransform_dev.parameters import configurations
from Featransform_dev.pipeline import Featransform
from atlantic.analysis import Analysis
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=Warning)
import sys
sys.path.append('C:/Users/utilizador/Desktop/GitHub/env/Packages_Dev')
#sys.path.append('C:/Users/luisf/OneDrive/Ambiente de Trabalho/GitHub/env/Packages_Dev')

##########   Create folder packaging
##########   Search more Anomaly detectors
##########   Test elementarly each feat sel method

######################################################  Load Data

data, target, prediction_type = load_dataset(dataset_selection='Titanic')
# 'Faceit', 'Healthcare', 'Spotify', 'Titanic'
# 'Textil', 'Cars_Prices', 'Fuel_Cars'

######################################################  Data Processing
####### Split Data

dp = Analysis(target=target)
data.isna().sum()
data.dtypes

############################################################################################################

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True)
train.isna().sum()

configs = configurations()
print(configs)

configs['Unsupervised']['Isolation_Forest']['n_estimators'] = 300
configs['DimensionalityReduction']['UMAP']['n_components'] = 6
configs['Clustering']['KMeans']['n_clusters'] = 3

ft = Featransform(validation_split = 0.30,
                  optimize_iters = 10,
                  configs = configs)

ft.fit_engineering(X = train, 
                   target = target)

train_ft = ft.transform(train)
test_ft = ft.transform(test)


### Save Featransform Metadata 
import pickle 
output = open("ft_eng.pkl", 'wb')
pickle.dump(ft, output)



