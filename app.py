from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('RFR.pkl')
onehot = joblib.load('five_joblib')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	c = ["make","model","vehicle_class","transmission","fuel","engine","cylinder","co2","co2Rating","smokerating"]
	df = pd.DataFrame(int_features,columns=c)
	l = onehot.transform(df.iloc[:,:5])
	c = onehot.get_feature_names_out()
	t = pd.DataFrame(l,columns=c)
	l2 = df.iloc[:,5:]
	final =pd.concat([l2,t],axis=1)
	result = model.predict(final)
	print("The Result is :",result)


	print(int_features)

	return render_template("main.html",prediction_text=" The Estimated Fuel consumption Rating is: {}".format(result))


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=6000, debug=True)
