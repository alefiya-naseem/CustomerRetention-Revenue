from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS
import pandas as pd
from model_wrapper import ModelWrapper
from scipy.stats import zscore

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api = Api(app)

data = pd.read_csv("df_holdout_scholarjet.csv")
data = pd.get_dummies(data).fillna(data.mean())

chi_means = pd.read_csv("chi_means.csv")
chi_means.drop("dayssinceenrollment", axis=1)
chi_stds = pd.read_csv("chi_std.csv")
chi_stds.drop("dayssinceenrollment", axis=1)

cols = list(chi_means.columns)

message = {}

message["dayssincelastord"] = "Days since last order is far below the expected value"
message["percdirtythirty"] = "The percentage of dirty orders in the last thirty days is far above the expected value"
message["numvisitthreeone"] = "Number of visits in the last three days is far below the expected value"
message["numvisitseventhree"] = "Number of visits in the last 3-7 days is far below the expected value"
message["numvisitthirtyseven"] = "Number of visits in the last 7-30 days is far below the expected value"
message["numvisitsixtythirty"] = "Number of visits in the last 30-60 days is far below the expected value"
message["numloggedinone"] = "Number of logins in the last one day is far below the expected value"
message["numloggedinthreeone"] = "Number of logins in the last 1-3 days is far below the expected value"
message["numloggedinseventhree"] = "Number of logins in the last 3-7 days is far below the expected value"
message["numloggedinthirtyseven"] = "'Number of logins in the last 7-30 days is far below the expected value"
message["numsecondsonsiteone"] = "Number of seconds on the site in the last 1 day is far below the expected value"
message["numsecondsonsiteseventhree"] = "Number of seconds on the site in the last 3-7 days is far below the expected value"
message["numsecondsonsitethirtyseven"] = "Number of seconds on the site in the last 7-30 days is far below the expected value"
message["numtotalpageviewsthirtyseven"] = "Number of total page views in the last 7-30 days is far below the expected value"
message["numatcone"] = "Number of items in cart for the last one day is far below the expected value"
message["numatcthreeone"] = "Number of items in cart for the last 1-3 days is far below the expected value"
message["numatcseventhree"] = "'Number of items in cart for the last 3-7 days is far below the expected value"
message["numatcthirtyseven"] = "Number of items in cart for the last  7-30 days is far below the expected value"
message["numideaboardseventhree"] = "Number of items in the favorites list within 3-7 days is far below the expected value"
message["dayssincelastvisit"] = "Days since last visit to the site is far below the expected value"
message["numsearchtermsthreeone"] = "Number of search terms in the last 1-3 days is far below the expected value"
message["numsearchtermsthirtyseven"] = "Number of search terms in the last 7-30 days is far below the expected value"
message["percsecondsinbound"] = " The percentage of call time that was inbound (customer called BAM) is far below the expected value"
message["percemailopenedone"] = "The percentage of emails (from Wayfair) from the past 1 day that were opened is far below the expected value"
message["percemailopenedthreeone"] = "The percentage of emails (from Wayfair) from the past 1-3 days that were opened is far below the expected value"
message["percemailopenedseventhree"] = "The percentage of emails (from Wayfair) from the past 3-7 days that were opened  is far below the expected value"
message["percemailopenedthirtyseven"] = "The percentage of emails (from Wayfair) from the past 7-30 days that were opened  is far below the expected value"
message["dayssinceenrollment"] = "Number of days since enrollment is far below the expected value"
message["roll_up_Unmanaged"] = "'roll_up_Unmanaged' is far below the expected value"
message["currentstatus_Enrolled"] = "'currentstatus_Enrolled' is far below the expected value"

def metrics(rec):
	rec = rec[cols]
	zs = (rec - chi_means.values)/chi_stds.values

	cols_bad = cols[:]						
	cols_bad.remove("percdirtythirty")

	zs_bad = zs[cols_bad] < -1.5
	zs_good = zs["percdirtythirty"] > 1.5
	zs_quality = pd.concat([zs_bad, zs_good], axis= 1).T
	cols_zs = list(zs_quality.index)
	zs_quality = [item for items in zs_quality.values for item in items]
	return "<br />".join([message[cols_zs[i]] for i in range(len(zs_quality)) if zs_quality[i]])

class Server(Resource):
	def get(self, id):
		m = ModelWrapper("pickle_model.pkl")

		customer = data[data.cuid == id]
	
		if len(customer) == 0:
			print("Customer not found")
			return {"conv": 0, "revenue": 0}

		print("Customer found")
		res = m.predict(customer)

		print("res[0]", res[0], "res[1]", res[1])

		mes = metrics(customer)


		return {"conv": res[0][0], "revenue": str(res[1].values[0]), "message": mes}

api.add_resource(Server, '/api/<int:id>')

if __name__ == '__main__':
	app.run(debug=True)
