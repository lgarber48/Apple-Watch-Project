import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv

df_HR_MASTER = pd.read_csv('/Users/louisgarber/Documents/McMaster Biomedical Eng/Apple Watch Project/apple_health_export/Processed HR Data/HR_Data1.csv')

with open('/Users/louisgarber/Documents/McMaster Biomedical Eng/Apple Watch Project/apple_health_export/Processed HR Data/Activity_Duration.csv',newline='') as file:
        reader=csv.reader(file)
        Activity_Duration=list(reader)

#Convert Activity Duration from strings to integers
Activity_Duration = list(map(int, Activity_Duration[0]))

#Enter Running Perceived Exertion (RPE) from notes
#RPE = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,4,6,4,4,6,4,6,7,6,5]
RPE = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,4,6,4,4,6,4,6]

# Enter Running Workout Type (RWT) (0=Regular Run and 1=Intervals)
#RWT = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
RWT = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#Rename each column so it has the format: "RPE-RWT" -> for example run #9: 4-0
for i in range(len(RPE)):
    df_HR_MASTER.rename(columns={df_HR_MASTER.columns[i]:str(RPE[i])+'-'+str(RWT[i])},inplace=True)

#Drop columns that contain 'nan' in the column title
df_HR_MASTER.drop(list(df_HR_MASTER.filter(regex = 'nan')), axis = 1, inplace = True)

#Find the average of each row (for each RPE-RWT) then plot
#Currently k = 4,6
for k in range(4,7,2):
    df_HR_MASTER[str(k)+'-0 mean']= df_HR_MASTER[str(k)+'-0'].mean(axis=1)

print(df_HR_MASTER.head(15))
print(Activity_Duration[0])

#PLOTTING
#Length of max workout duration (in s) spaced out at intervals of 120 (s)
xlength=len(range(0,max(Activity_Duration),120))
#Length of df with all the HRs from selected workouts
df_len=len(df_HR_MASTER)

#Equation to ensure the correct # of xticks are created to space intervals of 120s
Diff=(round(df_len/xlength)*xlength)-df_len

x=range(0,df_len+Diff,round(df_len/xlength))
xlabel=range(0,max(Activity_Duration),120)

ax=df_HR_MASTER['4-0 mean'].plot(color='blue',label='4-0 mean')
ax.set_xticks(x)
ax.set_xticklabels([x for x in xlabel])
plt.xlabel('Time (s)')
plt.ylabel('Beats Per Minute (BPM)')
plt.title('Heart Rate Data for Selected Workouts')

df_HR_MASTER['6-0 mean'].plot(ax=ax, color='red',label='6-0 mean')

ax.legend()
plt.show()


#RPE estimator 
#Over time build up enough data with RPE and then find the mean heart rate for each rank (i.e heart rate at 2,3,4 etc)
#Create ranges for each to assign an RPE given an HR
#For example lets say that on average an PRE of 4 has a mean HR of 120 and PRE of 5 a mean HR of 127 then PRE = 4 if HR >120 and < 127

#Workout type predictor
#Play around with RPE-1 data
#Figure out how to teach the model to learn the features of an RPE-1 (interval) workout



