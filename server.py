
from flask import Flask, request
from flask_restful import Resource, Api
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix , classification_report
import warnings
warnings.filterwarnings('ignore')





train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
# Train_test_split
X = train['text']
Y = train['target']
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.30, shuffle = True, random_state = 1)
test_split = test['text']
# Extracting Features
cv = CountVectorizer()
features = cv.fit_transform(x_train)
# Building Model
class_weight = {
    0 : 1,
    1 : 2
}
model = svm.SVC(kernel = 'rbf',class_weight = class_weight)
model.fit(features, y_train)


app = Flask(__name__)
api = Api(app)
class Classify(Resource):
    def post(self): 
        data = request.get_json() 
       
        # model = joblib.load('model3.pkl') # loading the model from disk
        if data.get("text"):
            transformed_data = cv.transform([data["text"]])
            res=model.predict(transformed_data)
            # result = '{ "result":'+(res)+'}'
            # jsonRes = json.loads(result)
            print(">>>>>>>>>>>>>>>>>>>>")
            
            result = json.dumps({"result": str(res[0])}, default=str) 
        else:
            result = json.dumps({"result": "No Content to classify !! "}, default=str) 
 
       
        print(result)

        return  result # returning the result of classification in JSON format



api.add_resource(Classify, '/classify')

app.run(port=5000)