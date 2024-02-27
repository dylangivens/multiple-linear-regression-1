import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option('display.max_columns', 200)

#read in data
df = pd.read_csv('C:/Users/dtgiv/Downloads/test_scores.csv')
print(df.columns)
print(df)

#exploratory data analysis
print(df.school.unique())
#23 distinct schools

print(df.school_setting.unique())
#Urban, Suburban, Rural

print(df.school_type.unique())
#Public, Non-public

print(df.classroom.unique())
#97 unique classrooms

print(df.teaching_method.unique())
#Standard, Experimental

print(df.n_student.unique())
#set histogram formatting to seaborn default
sns.set()
bin_edges = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
df.hist(column = 'n_student', bins = bin_edges)
plt.title('Class Size Distribution')
plt.xlabel('Class Size')
plt.ylabel('Count')
plt.show()

n = len(pd.unique(df['student_id']))
print(n)
#table grain is student_id

print(df.gender.unique())

print(df.lunch.unique())

print(df['pretest'].describe())
print(df['posttest'].describe())

#check for nulls
print(df.isnull().sum())

#change school to category
df['school'] = df['school'].astype('category')
df['school'] = df['school'].cat.codes
print(df.school.unique())

#change school_setting to category
df['school_setting'] = df['school_setting'].astype('category')
df['school_setting'] = df['school_setting'].cat.codes
print(df.school_setting.unique())

#change school_type to category
df['school_type'] = df['school_type'].astype('category')
df['school_type'] = df['school_type'].cat.codes
print(df.school_type.unique())

#change classroom to category
df['classroom'] = df['classroom'].astype('category')
df['classroom'] = df['classroom'].cat.codes
print(df.classroom.unique())

#change teaching_method to category
df['teaching_method'] = df['teaching_method'].astype('category')
df['teaching_method'] = df['teaching_method'].cat.codes
print(df.teaching_method.unique())

#change student_id to category
df['student_id'] = df['student_id'].astype('category')
df['student_id'] = df['student_id'].cat.codes
print(df.student_id.unique())

#change gender to category
df['gender'] = df['gender'].astype('category')
df['gender'] = df['gender'].cat.codes
print(df.gender.unique())

#change lunch to category
df['lunch'] = df['lunch'].astype('category')
df['lunch'] = df['lunch'].cat.codes
print(df.lunch.unique())

#view correlations with target variable
print(df.corr()['posttest'])

#drop variables with weak correlations
df = df.drop(['school', 'school_setting', 'classroom', 'student_id', 'gender'], axis = 1)
print(df)

#scatterplots exploring correlations between training features and target variable
plt.scatter(df['school_type'], df['posttest'])
plt.xlabel('School Type')
plt.ylabel('Post-Test Score')
plt.show()

plt.scatter(df['teaching_method'], df['posttest'])
plt.xlabel('Teaching Method')
plt.ylabel('Post-Test Score')
plt.show()

plt.scatter(df['n_student'], df['posttest'])
plt.xlabel('Students per Classroom')
plt.ylabel('Post-Test Score')
plt.show()

plt.scatter(df['lunch'], df['posttest'])
plt.xlabel('Free or Reduced Lunch')
plt.ylabel('Post-Test Score')
plt.show()

plt.scatter(df['pretest'], df['posttest'])
plt.xlabel('Pre-Test Score')
plt.ylabel('Post-Test Score')
plt.show()

#create training features dataset
x = df.drop(columns = 'posttest')
print(x)

y = df['posttest']
print(y)

#split x and y into training and test groups
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#train the model
lr = LinearRegression()
lr.fit(x_train, y_train)

#what's the y-int
c = lr.intercept_
print(c)

#what are the slopes
m = lr.coef_
print(m)

#test model
y_pred_train = lr.predict(x_train)
print(y_pred_train)

print(r2_score(y_train, y_pred_train))
#94.898% of the variance in posttest scores can be explained by my training features

y_pred_test = lr.predict(x_test)
print(y_pred_test)
print(r2_score(y_test, y_pred_test))
