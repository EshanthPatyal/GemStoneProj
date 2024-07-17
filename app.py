from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict import CustomData,PredictPipeline

application=Flask(__name__)

app=application
app.secret_key = 'your_secret_key' 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if(request.method=='GET'):
        return render_template('home.html')
    else:
        data=CustomData(
            carat=float(request.form.get('carat')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity'),
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        flash(f'The Predicted Score is {results[0]}')
        # return render_template('home.html',stri="The Predicted Score is",results=results[0])
        return redirect(url_for('predict_datapoint'))


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        