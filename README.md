<br>
<p align="center">
  <h2 align="center"> Featransform: Automated Feature Engineering Framework for Supervised Machine Learning
  <br>
  
## Framework Contextualization <a name = "ta"></a>

The `Featransform` project constitutes an objective and integrated proposition to automate feature engineering through the integration of various approachs of input pattern recognition known in Machine Learning such as dimensionality reduction, anomaly detection, clustering approaches and datetime feature constrution. This package provides an ensemble of diverse applications of each specific approach, aggregating and generating them all as added engineered features based on the original input columns. 

In order to avoid generation of noisy data for predictive consumption, after the engineered features ensemble are concatenated with the original features, a backwards wrapper feature selection also known as backward elimination is implemented to iteratively remove features based on evaluation of relevance, maintaining only valuable columns available for future models performance improvement purposes.

The architecture design includes three main sections, these being: data preprocessing, diverse feature engineering ensembles and optimized feature selection validation.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed feature engineering procedures are applicable on any data table associated with any Supervised ML scopes, based on input data columns to be built up on.
    
* Improvement of predictive results: The application of the `Featransform` aims at improve the predictive performance of future applied Machine Learning models through added feature construction, increased pattern recognition and optimization of existing input features.

* Continuous integration: After the train data is fitted, the created object can be saved and implemented in future data with the same structure. 
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 

* [Pandas](https://pandas.pydata.org/)
* [Sklearn](https://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/)
* [Optuna](https://optuna.org/)
    
## Where to get it <a name = "ta"></a>
    
Binary installer for the latest released version is available at the Python Package Index [(PyPI)](https://pypi.org/project/featransform/).   

## Installation  

To install this package from Pypi repository run the following command:

```
pip install featransform
```

# Usage Example
    
## Featransform - Automated Feature Engineering Pipeline

In order to be able to apply the automated feature engineering `featransform` pipeline you need first to import the package. 
The following needed step is to load a dataset and define your to be predicted target column name into the variable `target`.
You can customize the `fit_engineering` method by altering the following running pipeline parameters:
* configs: Nested dictionary in which are contained all methods specific parameters configurations. Feel free to customize each method as you see fit (customization example shown bellow);
* optimize_iters: Number of iterations generated for backwards feature selection optimization.
* validation_split: Division ratio in which the feature engineering methods will be evaluated within the loaded Dataset (range: [0.05, 0.45]).



Relevant Note:
* Although functional, `Featransform` pipeline is not optimized for big data purposes yet.

```py
    
import pandas as pd
from sklearn.model_selection import train_test_split
from featransform.pipeline import (Featransform,
                                   configurations)
import warnings
warnings.filterwarnings("ignore", category=Warning) # -> For a clean console
    
data = pd.read_csv('csv_directory_path', encoding='latin', delimiter=',') # Dataframe Loading Example

train,test = train_test_split(data, train_size=0.8)
train,test = train.reset_index(drop=True), test.reset_index(drop=True) # -> Required 


# Load and Customize Parameters

configs = configurations()
print(configs)

configs['Unsupervised']['Isolation_Forest']['n_estimators'] = 300
configs['Clustering']['KMeans']['n_clusters'] = 3
configs['DimensionalityReduction']['UMAP']['n_components'] = 6

## Fit Data

ft = Featransform(configs = configs,        # validation_split:float, optimize_iters:int 
                  optimize_iters = 10,
                  validation_split = 0.30) 

ft.fit_engineering(X = train,              # X:pd.DataFrame, target:str="Target_Column"
                   target = "Target_Column_Name")

## Transform Data 

train = ft.transform(X=train)
test = ft.transform(X=test)

# Export Featransform Metadata

import pickle
output = open("ft_eng.pkl", 'wb')
pickle.dump(ft, output)
    
```  

#### Further Implementations

Further automated and customizable feature engineering ensemble methods applications can be checked here: [Featransform Examples](https://github.com/TsLu1s/Featransform/tree/main/examples)

## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/Featransform/blob/main/LICENSE) for more information.

## Contact 
 
[Luis Santos - LinkedIn](https://www.linkedin.com/in/lu%C3%ADsfssantos/)

