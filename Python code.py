import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("/Users/ashikshafi/Downloads/google-play-store-apps/googleplaystore.csv")
df.Rating.describe()
#deleting an outlier from Ratings
df.iloc[10472,2]=np.NaN
df.boxplot("Rating")

#Deleting string characters from Installs and converting it to integers
df.Installs = df.Installs.apply(lambda x: x.replace(',',''))
df.Installs = df.Installs.apply(lambda x: x.replace('+',''))
df.Installs = df.Installs.apply(lambda x: x.replace('Free', '0'))
df.Installs = df.Installs.apply(lambda x: int(x))


Sorted_value = sorted(list(df['Installs'].unique()))
print(Sorted_value)
df['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace = True )

print(df["Installs"].head(20))
plt.figure(figsize = (10,10))
sns.regplot(x="Installs", y="Rating", color = 'blue',data=df);
plt.title('Rating VS Installs',size = 15)

#Fixing Size

df.Size = df.Size.apply(lambda x: x.replace('M',''))
df.Size = df.Size.apply(lambda x: x.replace('k',''))
df.Size = df.Size.apply(lambda x: x.replace('+',''))
df.Size = df.Size.apply(lambda x: x.replace(',',''))

df.Size = df.Size.apply(lambda x: x.replace('Varies with device', "NAN"))

df.Size = df.Size.apply(lambda x: float(x))

df.Size.describe()
df.boxplot("Size")
plt.hist(df["Size"], bins=30)


#Now to last updated

df["Last Updated"].head(20)

print(sorted(df["Last Updated"])[1:3])
df["Last Updated"] = df["Last Updated"].apply(lambda x: x.replace('1.0.19', "NAN"))
df["Last Updated"] = df["Last Updated"].apply(lambda x: x.replace('NAN', "April 1, 2016"))

from datetime import datetime

Today= datetime.strptime("October 7, 2018", '%B %d, %Y')
UpdateDate = list(map(lambda x: datetime.strptime(x, '%B %d, %Y'), df["Last Updated"]))

UpdateDate[0:5]

UpdateTime=[]
for x in UpdateDate:
    UpdateTime.append(Today-x)
print(UpdateTime[0:4])

#Getting the number of days since last update
df["Last Updated"]=pd.to_timedelta(UpdateTime)
df["Last Updated"]=df["Last Updated"].apply(lambda x: x.total_seconds()/86400)
#Getting the number of days in float object by taking seconds and dividing by number of second in a day.
df["Last Updated"].loc[30:35]



plt.hist(df["Last Updated"], bins=20)
plt.boxplot(df["Last Updated"])


#Data vizualization

Newdf=df[["Rating", "Size", "Installs", "Last Updated"]].copy()

sns.pairplot(Newdf, palette="husl")

corr=Newdf.corr()
fig, ax = plt.subplots(figsize=(4, 4))
colormap = sns.diverging_palette(220, 4, as_cmap=True)
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
# Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns, rotation=70)
# Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
# show plot
plt.show()


#Machine learing with regression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

Newdf=Newdf.reset_index()
Newdf=Newdf.dropna(axis='index')

X=Newdf[["Size", "Installs", "Last Updated"]].values.view()

X.reshape(-1,3)

y=Newdf[["Rating"]].values.view()
y.reshape(-1,1)

#Linear regression


reg = linear_model.LinearRegression()


reg.fit(X, y)
print(reg.intercept_)
print(reg.coef_)

y_pred = reg.predict(X)

print((y_pred)[:10])


print(reg.score(X, y))

plt.plot(X, y_pred, '.', color='teal', linewidth=2)

fig = plt.figure()
fig.suptitle("Predicting Rating with Size, Installs and Update period", fontsize=16)
plt.subplot(3, 1, 1)
plt.title("Size vs Rating")
sns.regplot(X[:,0], y_pred[:,0], color="teal" )
plt.subplot(3, 1, 2)
plt.title("Install vs. Rating")
sns.regplot(X[:,1], y_pred[:,0], color="red" )
plt.subplot(3, 1, 3)
plt.title("Last update vs Rating")
sns.regplot(X[:,2], y_pred[:,0], color="blue" )
plt.show()
plt.tight_layout()


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y[:,0],test_size = .3, random_state=42)

# Create the regressor: reg_all
reg_all = linear_model.LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Create a linear regression object: reg
reg_val = linear_model.LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg_val, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

#End of linear regression

# KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable


# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X_train, y_train)


# Predict the labels for the training data X
y_pred = knn.predict(X_train)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

