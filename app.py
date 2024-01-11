from flask import Flask, render_template,request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictPipeline



app = Flask(__name__)

@app.route('/',methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/train',methods=['GET'])
def trainpage():
    os.system("python main.py")
    return "Successfully trained"
@app.route('/predict',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            pclass =int(request.form['pclass'])
            nom =str(request.form['nom'])
            sexe =str(request.form['sexe'])
            age =int(request.form['age'])
            cabin_num =int(request.form['cabin_num'])
            free_sulfur_dioxide =float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide =float(request.form['total_sulfur_dioxide'])
            density =float(request.form['density'])
            pH =float(request.form['pH'])
            sulphates =float(request.form['sulphates'])
            alcohol =float(request.form['alcohol'])
       
         
            data = [pclass,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]
            data = np.array(data).reshape(1, 11)
            
            predict = PredictPipeline()
            predict = predict.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')


if __name__=="__main__":
    # app.run(host="0.0.0.0",port= 8080, debug=True)
    app.run(host="0.0.0.0",port= 8080)
