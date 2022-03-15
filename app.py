from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import gunicorn
from feature import generate_data_set
# Gradient Boosting Classifier Model
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv("phishing.csv")
#droping index column
data = data.drop(['Index'],axis = 1)
# Splitting the dataset into dependant and independant fetature

X = data.drop(["class"],axis =1)
y = data["class"]

# instantiate the model
gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# fit the model 
gbc.fit(X.values,y)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", xx= -1)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,30) 
        #1 is safe       
        #-1 is unsafe
        predicted_url = gbc.predict_proba(x)[0,1]
        return render_template('index.html',xx =round(predicted_url,2),url=url )
        
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)
       
        
        
        
