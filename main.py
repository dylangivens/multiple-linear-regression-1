import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 200)

#read in data
df = pd.read_csv('C:/Users/dtgiv/Downloads/test_scores.csv')
print(df.columns)
print(df)

#exploratory data analysis
print(df.school.unique())
#11 distinct schools

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

#follow video, need to change everything to numeric values