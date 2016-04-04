import pandas
from sklearn.preprocessing import LabelEncoder
from collections import namedtuple

def get_data():
	df = pandas.read_csv('../data/train.csv', header=0)
	le = LabelEncoder()

	df.workclass = le.fit_transform(df.workclass)
	df.education = le.fit_transform(df.education)
	df['marital-status'] = le.fit_transform(df['marital-status'])
	df.occupation = le.fit_transform(df.occupation)
	df.relationship = le.fit_transform(df.relationship)
	df.race = le.fit_transform(df.race)
	df.sex = le.fit_transform(df.sex)
	df['native-country'] = le.fit_transform(df['native-country'])
	df.income = le.fit_transform(df.income)

	data, target = df.values[:,:-1], df.values[:,-1:] # We split the class from the rest

	series = namedtuple('Series', ['data', 'target'])
	series.data = data
	series.target = target.ravel()
	
	return series
