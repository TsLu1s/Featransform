import pandas as pd
from atlantic.processing.analysis import Analysis 
from featransform.optimizer.selector import Selector
from featransform.configs.parameters import configurations
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error,
                             explained_variance_score,
                             max_error,
                             r2_score,
                             accuracy_score,
                             precision_score,
                             f1_score,
                             recall_score)
import xgboost as xgb
import optuna
from tqdm import tqdm
import warnings
import logging

sec_conf = configurations()

class Evaluation:
    def __init__(self, 
                 train : pd.DataFrame, 
                 test : pd.DataFrame, 
                 target : str, 
                 optimize_iters : int = 8,
                 configs : dict = sec_conf):
        """
        Initialize the Evaluation class.

        :param train: Training dataset.
        :param test: Testing dataset.
        :param target: Target column name.
        :param optimize_iters: Number of iterations for feature upgrading.
        :param configs: Configuration dictionary.
        """
        self._train = train
        self._test = test
        self.target = target
        self.optimize_iters = optimize_iters
        self.configs = configs
        self.metrics = None # Placeholder for final metrics
        self._tmetrics = None # Placeholder for temporary metrics
        self.va_imp = None # Placeholder for variable importances
        self.hparameters_list, self.metrics_list = [],[] # Lists to store hyperparameters and metrics
        self.performance_history, self.fs_metrics = [],[] # Lists for performance history and feature selection metrics
        self.dp = Analysis(target=target) # Analysis instance for target column properties
        
    def objective(self,trial): 
        """
        Objective method for Optuna optimization.

        :param trial: Optuna Trial object.
        :return: None
        """
        # Extract training and testing datasets based on target type 
        self.pred_type, self.eval_metric = self.dp.target_type(X=self._train)

        X_train, X_test, y_train, y_test = self.dp.divide_dfs(train=self._train,test=self._test)

        # Configure logging to suppress Optuna's logs
        logging.getLogger('optuna').setLevel(logging.CRITICAL)
        
        # Define the regression and classification models
        xgb_regressor, xgb_classifier = xgb.XGBRegressor(), xgb.XGBClassifier()
                    
        # Define hyperparameters for XGBoost regression
        xgb_regressor_params = {
            'n_estimators': trial.suggest_int('xgb_regressor_n_estimators', 40, 150),
            'max_depth': trial.suggest_int('xgb_regressor_max_depth', 5, 10),
            'learning_rate': trial.suggest_loguniform('xgb_regressor_learning_rate', 0.01, 0.1),
        }
        
        # Define hyperparameters for XGBoost classification
        xgb_classifier_params = {
            'n_estimators': trial.suggest_int('xgb_classifier_n_estimators', 40, 150),
            'max_depth': trial.suggest_int('xgb_classifier_max_depth', 5, 10),
            'learning_rate': trial.suggest_loguniform('xgb_classifier_learning_rate', 0.01, 0.1),  
        }
        # Suppress warnings during model training
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
                    
            if self.pred_type == 'Reg':
                # Initialize the regression models with the suggested hyperparameters
                xgb_regressor.set_params(**xgb_regressor_params)
                
                # Train the regression models on the training data
                xgb_regressor.fit(X_train, y_train)
                
                # Make predictions on the test data for regression models
                xgb_r_preds = xgb_regressor.predict(X_test)
                
                m_reg = pd.DataFrame(self.metrics_regression(y_test ,xgb_r_preds),index = [0])
                m_reg['Model'] = 'XGBoost'
                m_reg['iteration'] = len(self.metrics_list) + 1
                self.metrics_list.append(m_reg)
                
                self.hparameters_list.append({
                    'xgb_regressor_params': xgb_regressor_params,
                    'iteration': len(self.metrics_list) + 1,
                    })
                
            elif self.pred_type == 'Class':
                # Initialize the classification model with the suggested hyperparameters
                xgb_classifier.set_params(**xgb_classifier_params)
                
                # Train the classification model on the training data
                xgb_classifier.fit(X_train, y_train)

                # Make predictions on the test data for classification model
                xgb_c_preds = xgb_classifier.predict(X_test)
                
                m_class = pd.DataFrame(self.metrics_classification(y_test ,xgb_c_preds),index = [0])
                m_class['Model'] = 'XGBoost'
                m_class['iteration'] = len(self.metrics_list)+1
                self.metrics_list.append(m_class)
                
                self.hparameters_list.append({
                    'xgb_classifier_params': xgb_classifier_params,
                    'iteration': len(self.metrics_list)+1,
                    })
        
    def auto_evaluate(self):
        """
        Perform automated evaluation using Optuna optimization.

        :return: DataFrame with the best-performing metrics.
        """
        # Set Optuna logging verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Set optimization direction and metric based on target type
        self.pred_type, self.eval_metric = self.dp.target_type(X = self._train)
        
        if self.pred_type == 'Reg':
            direction_, study_name_, metric, ascending_ ='minimize', 'Reg Evaluation', 'Mean Absolute Error', True
        elif self.pred_type == 'Class':
            direction_, study_name_, metric, ascending_ ='maximize', 'Class Evaluation', 'Accuracy', False
        # Create an Optuna study
        study = optuna.create_study(direction = direction_, study_name = study_name_)
        # Optimize the objective function with tqdm progress bar
        with tqdm(total = 10, desc = '', ncols = 75) as pbar:
            def trial_callback(study, trial):
                pbar.update(1)
            study.optimize(lambda trial: self.objective(trial),
                                         n_trials = 10,
                                         callbacks = [trial_callback])
        
        # Concatenate metrics for all iterations and select the best-performing metric
        self.metrics = pd.concat(self.metrics_list)
        self.metrics = self.metrics.sort_values(['Model', metric], ascending = ascending_)
        self._tmetrics = self.metrics.copy()
        self.metrics = self.metrics.iloc[[0], :]
        del self.metrics['iteration']
        
        return self.metrics 

    def feature_upgrading(self):
        """
        Perform feature selection using Catboost for multiple iterations.

        :return: List of selected features for each iteration.
        """
        # Perform feature selection using Catboost
        selector = Selector(configs = self.configs)
        self.va_imp = selector.feature_selection(X = self._train, target = self.target)
        train, test = self._train.copy(), self._test.copy()
        # Filter selected columns based on the variable importance threshold
        thresholds = [i / 100 for i in range(100, 90, -1)]
        thresholds = thresholds[:self.optimize_iters]
        
        sel_features = [sorted(self.va_imp.loc[self.va_imp['Importances'].cumsum() <= threshold, 'Feature Id'].tolist() + [self.target],
                               reverse = True) for threshold in thresholds]
        self.va_imp.columns
        print('')
        for iteration in range(0,len(sel_features)):
            
            sel_cols = [*sel_features[iteration]]
            self._train, self._test = train[sel_cols], test[sel_cols]
            
            print('Iteration', iteration + 1, '|| Total Features:', len(sel_cols))
            
            # Evaluate the performance using the auto_evaluate method
            metrics = self.auto_evaluate()
            metrics['iteration'] = iteration
            print('Performance :', round(metrics.iloc[0][0],4))
            print('')
            
            self.fs_metrics.append(metrics)
            # Save the performance metrics for comparison
            self.performance_history.append({
                'selected_cols': sel_cols,
                'iteration': iteration,
                'performance': round(metrics.iloc[0][0],4)})
        
        # Determine the best iteration and referring columns based on performance
        if self.pred_type == 'Class':
            _, best_iteration_cols = max(
                (entry['iteration'], entry['selected_cols']) for entry in self.performance_history
                if entry['performance'] == max(self.performance_history, key=lambda x: x['performance'])['performance'])
        elif self.pred_type == 'Reg':
            _, best_iteration_cols = max(
                (entry['iteration'], entry['selected_cols']) for entry in self.performance_history
                if entry['performance'] == min(self.performance_history, key=lambda x: x['performance'])['performance'])

        return best_iteration_cols

    @staticmethod
    def metrics_regression(y_true, y_pred):
        # Calculate various regression prediction evaluation metrics.
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
        maximo_error = max_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics_reg = {'Mean Absolute Error' : mae,
                       'Mean Absolute Percentage Error' : mape,
                       'Mean Squared Error' : mse,
                       'Explained Variance Score' : evs,
                       'Max Error': maximo_error,
                       'R2 Score' : r2}
        
        return metrics_reg
    
    @staticmethod
    def metrics_classification(y_true, y_pred):
        # Calculate various classification prediction evaluation metrics.
        accuracy_metric = accuracy_score(y_true, y_pred)
        precision_metric = precision_score(y_true, y_pred,average='micro')
        f1_macro_metric = f1_score(y_true, y_pred,average='macro')
        recall_score_metric = recall_score(y_true, y_pred, average='macro')
        
        metrics_class = {'Accuracy': accuracy_metric,
                         'Precision Micro': precision_metric,
                         'F1 Score Macro' : f1_macro_metric,
                         'Recall Score Macro' : recall_score_metric}
        
        return metrics_class

