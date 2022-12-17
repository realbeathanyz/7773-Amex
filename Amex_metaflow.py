"""

Simple stand-alone script showing end-to-end training of a regression model using Metaflow. 
This script ports the composable script into an explicit dependency graphg (using Metaflow syntax)
and highlights the advantages of doing so. This script has been created for pedagogical purposes, 
and it does NOT necessarely reflect all best practices.

Please refer to the slides and our discussion for further context.

MAKE SURE TO RUN THIS WITH METAFLOW LOCAL FIRST

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import comet_ml
from comet_ml import Experiment
from comet_ml import init
from comet_ml.integration.metaflow import comet_flow
from metaflow import FlowSpec, step, Parameter, IncludeFile, current

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import os

from datetime import datetime

#assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']
#assert 'MY_PROJECT_NAME' in os.environ and os.environ['MY_PROJECT_NAME']
#print("Running experiment for project: {}".format(os.environ['MY_PROJECT_NAME']))


from comet_ml import Experiment
exp = Experiment(
                api_key=os.environ['dZ7uQgPG1JZyOLPdSDPJzZp5V'],
                project_name=os.environ['final-project-amex'],
                auto_param_logging=False
                )


# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


class LGBM(FlowSpec):
    """
    SampleRegressionFlow is a minimal DAG showcasing reading data from a file 
    and training a model successfully.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files

    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
    )

    VALIDATION_SPLIT = Parameter(
        name='validation_split',
        help='Determining the split of the dataset for validating',
        default=0.20)
    


    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self): 
        self.train_data = pd.read_feather('./train_data.ftr')
        self.test_data = pd.read_feather('./test_data.ftr')
        self.train_label = pd.read_csv('./train_labels.csv')
        self.train_df = self.train_data.merge(self.train_label, left_on='customer_ID', right_on='customer_ID')

        
        self.next(self.prepare_train_and_test_dataset)
        
    @step
    def prepare_train_and_test_dataset(self):
        self.train_df = self.train_data.merge(self.train_label, left_on='customer_ID', right_on='customer_ID')
        df = self.train_df.dropna(axis = 1, thresh = 0.7 * self.train_data.shape[0])
        
        y = df[["target"]]
        X = df.drop(["customer_ID", "S_2", "month", "target"], axis = 1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.15,  stratify=y)

        print("the shape of our training data is " + str(self.X_train.shape))
        print("the shape of our test data is " + str(self.X_test.shape))
        
        
        self.next(self.train_model)


        
    @step
    def train_model(self):
        LGBM = lgb.LGBMClassifier(boosting_type='goss', 
                                  max_depth=20, random_state=0, 
                                  n_estimators=200, 
                                  learning_rate=0.09, 
                                  num_leaves=500)
        LGBM = LGBM.fit(self.X_train, self.y_train)
        train_score = LGBM.score(self.X_train, self.y_train)

        print("the score of our LGBM model on the training data is " + str(train_score))
        
        test_score = LGBM.score(self.X_test, self.y_test)

        print("the score of our LGBM model on the test data is " + str(test_score))
        self.next(self.predict_model)




    @step 
    def predict_model(self):
        columns = self.X.columns.tolist()
        
        test_df = self.test_data[columns]
        
        self.probs = LGBM.predict_proba(test_df)
        self.probs
        
        self.probs[:, 1]
        
        pred = LGBM.predict(test_df)
        pred

        self.next(self.end)
        
    @step
    def end(self):
        # all done, just print goodbye
        result= pd.DataFrame()
        result["customer_ID"] = self.test_data["customer_ID"]
        result["prediction"] = self.probs[:, 1]
        result.to_csv('result.csv', index=None)
        
        result
if __name__ == '__main__':
    LGBM()

