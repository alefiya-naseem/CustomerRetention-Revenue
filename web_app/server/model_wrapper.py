import pickle
from model import Model

class ModelWrapper(object): 	
	def __init__(self, path):
		self.featureList = ['dayssincelastord',  'percdirtythirty',  'numvisitthreeone',  'numvisitseventhree',  'numvisitthirtyseven',  'numvisitsixtythirty',  'numloggedinone',  'numloggedinthreeone',  'numloggedinseventhree',  'numloggedinthirtyseven',  'numsecondsonsiteone',  'numsecondsonsiteseventhree',  'numsecondsonsitethirtyseven',  'numtotalpageviewsthirtyseven',  'numatcone',  'numatcthreeone',  'numatcseventhree',  'numatcthirtyseven',  'numideaboardseventhree',  'dayssincelastvisit',  'numsearchtermsthreeone',  'numsearchtermsthirtyseven',  'percsecondsinbound',  'percemailopenedone',  'percemailopenedthreeone',  'percemailopenedseventhree',  'percemailopenedthirtyseven',  'dayssinceenrollment',  'roll_up_Unmanaged',  'currentstatus_Enrolled']

		self.clf = pickle.load(open(path, 'rb'))

	def predict(self, X):
		X = X[self.featureList]
		pre = self.clf.predict(X)
		return pre