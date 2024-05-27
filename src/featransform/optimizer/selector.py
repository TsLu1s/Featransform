import pandas as pd
from atlantic.processing.analysis import Analysis
from featransform.configs.parameters import configurations
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb

sec_conf = configurations()

class Selector:
    def __init__(self,
                 configs : dict = sec_conf):
        """
        Initialize the Selector class.

        :param configs: Configuration dictionary for feature selection.
        """
        self.configs = configs
        self.sel_cols = [] # Placeholder for selected columns
        self.feat_importance = None # Placeholder for feature importances
        self.ftimp_cb = None # Placeholder for CatBoost feature importances
        self.ftimp_xgb = None # Placeholder for LabelEncoder instance

    def feature_selection(self, 
                          X : pd.DataFrame,
                          target : str):
        """
        Perform feature selection using CatBoost & XGBoost combination for feature importances.

        :param X: Input DataFrame containing features.
        :param target: Name of the target column for classification or regression.
        :return: DataFrame with selected columns and their feature importances.
        """
        # Copy the input DataFrame to avoid modifying the original data
        X_ = X.copy()
        
        # Extract input columns excluding the target column
        input_cols = [col for col in X_.columns if col != target]

        # Determine the task type (classification or regression).
        dp = Analysis(target = target)
        pred_type, _ = dp.target_type(X = X_)
        
        # Extract feature selection parameters from the configurations
        fs_keys = list(self.configs['FeatureSelection'].keys())
        cb_params = self.configs['FeatureSelection']['CatBoost']
        xgb_params = self.configs['FeatureSelection']['XGBoost']
        
        # Classification Use Case
        if pred_type == 'Class':
            # Encode target column
            self._label_encoder = LabelEncoder()
            X_[target] = self._label_encoder.fit_transform(X_[target].copy())
            
            # CatBoost feature importance
            if 'CatBoost' in fs_keys:
                if len(X_[target].unique()) == 2:  # Binary classification
                    model = CatBoostClassifier(**cb_params,
                                               loss_function = 'Logloss')
                else:  # Multi-class classification
                    model = CatBoostClassifier(**cb_params,
                                               loss_function = 'MultiClass')
                model.fit(X_[input_cols], X_[target])
                # Get feature importances from the trained CatBoost model.
                self.ftimp_cb = model.get_feature_importance(prettified = True)
                self.ftimp_cb['Importances'] = self.ftimp_cb['Importances']/100
            # XGBoost feature importance
            if 'XGBoost' in fs_keys:
                model_xgb = xgb.XGBClassifier(**xgb_params)
                model_xgb.fit(X_[input_cols], X_[target])
                # Get feature importances from the trained XGBoost model.
                self.ftimp_xgb = pd.DataFrame({'Feature Id': input_cols, 'Importances': model_xgb.feature_importances_}) \
                                            .sort_values(by = 'Importances', ascending = False)
        else: # Regression
            # CatBoost feature importance
            if 'CatBoost' in fs_keys:
                model = CatBoostRegressor(**cb_params,
                                          loss_function = 'MAE')
                model.fit(X_[input_cols], X_[target])
                # Get feature importances from the trained CatBoost model.
                self.ftimp_cb = model.get_feature_importance(prettified = True)
                self.ftimp_cb['Importances'] = self.ftimp_cb['Importances']/100
            # XGBoost feature importance
            if 'XGBoost' in fs_keys: 
                model_xgb = xgb.XGBRegressor(**xgb_params)
                model_xgb.fit(X_[input_cols], X_[target])
                # Get feature importances from the trained XGBoost model.
                self.ftimp_xgb = pd.DataFrame({'Feature Id': input_cols, 'Importances': model_xgb.feature_importances_}) \
                                            .sort_values(by = 'Importances', ascending = False)

        # Combine feature importances if both models are used                 
        if self.ftimp_cb is None:
            self.feat_importance = self.ftimp_xgb
        elif self.ftimp_xgb is None:
            self.feat_importance = self.ftimp_cb
        else:    
            self.feat_importance = pd.concat([self.ftimp_cb, self.ftimp_xgb]) \
                                     .groupby('Feature Id')['Importances'].sum().reset_index() \
                                     .sort_values(by='Importances', ascending=False)
            self.feat_importance['Importances'] = self.feat_importance['Importances']/2
            
        # Return the selected columns and their feature importances.
        return self.feat_importance


