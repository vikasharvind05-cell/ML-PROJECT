import joblib
import pandas as pd
import seaborn as sns
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
#Load Dataset
#-------------------

#df=pd.read_csv("data/Titanic.csv")
df=sns.load_dataset("titanic")

#keep only required column
df=df[["pclass","sex","age","fare","survived"]]

x=df.drop("survived",axis=1)
y=df["survived"]

numeric_features=["age","fare"]
categorical_features=["sex","pclass"]

#---------------
#numerical pipeline
#--------------

numeric_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])

#---------------
#categorical Pipeline
#-----------------

categorical_pipeline=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OrdinalEncoder())])

#--------------
#combine
#------------

preprocessor=ColumnTransformer([("num",numeric_pipeline,numeric_features),("cat",categorical_pipeline,categorical_features)])

#---------------
#final pipeline
#-------------

pipeline=Pipeline([("preprocessor",preprocessor),("classifier",LogisticRegression())])
pipeline.fit(x,y)

#create the model directory if it doesn't exit
os.makedirs("model",exist_ok=True)
joblib.dump(pipeline,"model/pipeline.pkl")
print("Pipeline saved successfully")