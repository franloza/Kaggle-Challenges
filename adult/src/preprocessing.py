import pandas
from sklearn.preprocessing import LabelEncoder

def get_data():
	df = pandas.read_csv('../data/train.csv', header=0)
	le = LabelEncoder()

	df.workclass = le.fit_transform(df.workclass)
	df.education = le.fit_transform(df.education)
	df.marital_status = le.fit_transform(df['marital-status'])
	df.occupation = le.fit_transform(df.occupation)
	df.relationship = le.fit_transform(df.relationship)
	df.race = le.fit_transform(df.race)
	df.sex = le.fit_transform(df.sex)
	df.native_country = le.fit_transform(df['native-country'])
	df = df.drop(['marital-status', 'native-country'],axis=1)

	return df
