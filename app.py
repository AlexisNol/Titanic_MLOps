from flask import Flask, render_template,request
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('c:/users/m.lemrabott/desktop/m2/cloud-computing/titanic_mlops/titanic_mlops-main/src')
from src.mlProject.pipeline.prediction import PredictPipelineV2
#from mlProject.pipeline.prediction import PredictPipelineV2 
#import sys
#sys.path.append('src/mlProject/pipeline/prediction.py')
#import PredictTitanic
#from mlProject.pipeline.prediction import PreditTitanic



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
            sibsp =int(request.form['sibsp'])
            parch =int(request.form['parch'])
       
         
            data = [pclass,sexe,sibsp,parch]
            #data2 = [pclass,sexe,sibsp,parch]
            data2 = np.array(data).reshape(1, 4)
            #data2 = np.array([1,2,3,4]).reshape(1, 4)
            
            predict = PredictPipelineV2()
            predict = predict.predict(data2)

            #return render_template('results.html')
            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')


if __name__=="__main__":
    # app.run(host="0.0.0.0",port= 8080, debug=True)
    #os.system("python main.py")
    app.run(host="0.0.0.0",port= 8080)
