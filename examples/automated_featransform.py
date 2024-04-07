from featransform.pipeline import (Featransform,
                                   AutoLabelEncoder, 
                                   AutoIterativeImputer,
                                   configurations)
from atlantic.analysis import Analysis
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=Warning)

######################################################  Load Data

# source_data = "https://www.kaggle.com/datasets/surekharamireddy/fraudulent-claim-on-cars-physical-damage"

url = "https://raw.githubusercontent.com/TsLu1s/Featransform/main/data/Fraudulent_Claim_Cars_class.csv"
data = pd.read_csv(url) # Dataframe Loading Example

target="fraud"
data[target]=data[target].astype('category')

###################################################### 

dp = Analysis(target=target)
data.isna().sum()
data.dtypes

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True)
train.isna().sum()

######################################################

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



