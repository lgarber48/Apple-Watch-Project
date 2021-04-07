import pandas as pd 
from matplotlib import pyplot as plt
import xmltodict
import datetime
import numpy as np

#Step 1: Automatically extracted and pre-processed the desired HR data from XML file into a organized and labeled dataframe
#Step 2: Apply Moving Average algorithm to smooth HR data

input_path='/Users/louisgarber/Documents/McMaster Biomedical Eng/Apple Watch Project/apple_health_export/export.xml'

with open(input_path,'r') as xml_file:
    input_data=xmltodict.parse(xml_file.read())

#Records list of general health data into a Pandas frame
record_list=input_data['HealthData']['Record']
df_records=pd.DataFrame(record_list)

#Workout data into Pandas frame
workout_list=input_data['HealthData']['Workout']
df_workouts=pd.DataFrame(workout_list)

print('Length of df_workouts: '+str(len(df_workouts)))

#Extract Workout Information
#Note: Only 6 total workouts have been uploaded (0-5)
def extract_workout(i):
    #Change some workout data from string to numeric (either float64 or int64)
    df_workouts[['@duration','@totalDistance','@totalEnergyBurned']]=df_workouts[['@duration','@totalDistance','@totalEnergyBurned']].apply(pd.to_numeric)

    #Convert dates to datetime format
    format ='%Y-%m-%d %H:%M:%S %z'
    df_workouts['@creationDate']=pd.to_datetime(df_workouts['@creationDate'],format=format)
    df_workouts['@startDate']=pd.to_datetime(df_workouts['@startDate'],format=format)
    df_workouts['@endDate']=pd.to_datetime(df_workouts['@endDate'],format=format)

    #Rename Activities (MAY NEED TO ADD NEW ACTIVITIES)
    df_workouts['@workoutActivityType']=df_workouts['@workoutActivityType'].replace('HKWorkoutActivityTypeWalking','Walking')
    df_workouts['@workoutActivityType']=df_workouts['@workoutActivityType'].replace('HKWorkoutActivityTypeFunctionalStrengthTraining','Strength')
    df_workouts['@workoutActivityType']=df_workouts['@workoutActivityType'].replace('HKWorkoutActivityTypeRunning','Running')
    
    #Save workout start and end times & workout type
    Start_time=df_workouts['@startDate'].iloc[i]
    End_time=df_workouts['@endDate'].iloc[i]
    Total_Time=End_time-Start_time
    Workout_Type=df_workouts['@workoutActivityType'].iloc[i]

    #Convert workout duration to seconds (from TimeDelta)
    Time_Comp=Total_Time.components
    Wk_Duration_Seconds=Time_Comp[0]*86400 + Time_Comp[1]*3600+ Time_Comp[2]*60 + Time_Comp[3] + Time_Comp[4]/1000
    #Convert to Workout Duration to int
    Wk_Duration_Seconds=int(Wk_Duration_Seconds)

    print('Workout Type: '+str(Workout_Type))
    print('Workout Start Time: '+str(Start_time))
    print('Workout End Time: '+str(End_time))  
    print('Workout Duration: '+str(Total_Time))
    print('Total Workout Duration in Seconds: '+str(Wk_Duration_Seconds))
    return Start_time, End_time, Wk_Duration_Seconds, Workout_Type

    #Save as CSV
    #finalADS=df_workouts.to_csv('/Users/louisgarber/Documents/McMaster Biomedical Eng/Apple Watch Project/apple_health_export/finalADS.csv',header=True)

#Extract Heart Rate over Workout Duration
def extract_HR(Wk_Data):
    
    #Select subset of data that contains HR data (.copy() avoids chaining issues)
    df_records_HR=df_records[df_records['@type']=='HKQuantityTypeIdentifierHeartRate'].copy()
    
    #Convert dates to datetime
    format ='%Y-%m-%d %H:%M:%S %z'
    df_records_HR['@creationDate']=pd.to_datetime(df_records_HR['@creationDate'],format=format)
    df_records_HR['@startDate']=pd.to_datetime(df_records_HR['@startDate'],format=format)
    df_records_HR['@endDate']=pd.to_datetime(df_records_HR['@endDate'],format=format)

    #Extract HR values during desired workout duration
    df_records_HR_Wk=df_records_HR[(df_records_HR['@startDate'] >= Wk_Data[0]) & (df_records_HR['@endDate'] <= Wk_Data[1])].copy()
    df_records_HR_Wk.reset_index(inplace=True)

    #Convert values to numeric
    df_records_HR_Wk['@value']=df_records_HR_Wk['@value'].apply(pd.to_numeric)

    #Average Heart Rate During Workout
    Avg_HR=round(df_records_HR_Wk['@value'].mean(),2)
    print('Avg. Heart Rate: '+str(Avg_HR)+' BPM')

    #Apply Moving Average -> Figure out Rolling Window
    df_records_HR_Wk['@value']=df_records_HR_Wk['@value'].rolling(7).mean()

    #Raname column to match activity 
    df_records_HR_Wk.rename(columns={'@value':Wk_Data[3]},inplace=True)

    return df_records_HR_Wk[Wk_Data[3]]

#Call functions
Wk_Data=[]
Wk_Duration=[]
Wk_Type=[]

#Iterate through all the workouts and save key information (Type, Start & End Times, Total Duration) into Wk_Data
for i in range(len(df_workouts)):
    Wk_Data.append(extract_workout(i))

#Save Workout duration and types into lists to reference when training the model (saved into Wk_Type)
for j in range(len(df_workouts)):
    Wk_List=Wk_Data[j]
    Wk_Duration.append(Wk_List[2])
    Wk_Type.append(Wk_List[3])

Wk_Duration=np.array(Wk_Duration)

#Extract HR data for all the selected workouts and save into 'HR MASTER' df
df_HR_MASTER=[]
 
#SELECT DESIRED WORKOUT TO EXTRACT AND PLOT (Select 'All' for all workouts)
Workout='All'

#Extract desired location of selected workout
Location=[i for i,x in enumerate(Wk_Type) if x == Workout]

if Workout == 'All':
    for k in range(len(df_workouts)): 
        data=extract_HR(Wk_Data[k])
        df_HR_MASTER.append(data)
    Activity_Duration = Wk_Duration
else:
    for k in Location: 
        data=extract_HR(Wk_Data[k])
        df_HR_MASTER.append(data)
    Activity_Duration = list(Wk_Duration[Location])
    

df_HR_MASTER=pd.concat(df_HR_MASTER, axis=1)

#Print Master Data Frame
print(df_HR_MASTER)
print(Activity_Duration)

#PLOTTING
#Create x range for plot
#Length of max workout duration (in s) spaced out at intervals of 120 (s)
xlength=len(range(0,max(Activity_Duration),120))
#Length of df with all the HRs from selected workouts
df_len=len(df_HR_MASTER)

#Equation to ensure the correct # of xticks are created to space intervals of 120s
Diff=(round(df_len/xlength)*xlength)-df_len

x=range(0,df_len+Diff,round(df_len/xlength))
xlabel=range(0,max(Activity_Duration),120)

ax=df_HR_MASTER.plot()
ax.set_xticks(x)
ax.set_xticklabels([x for x in xlabel])
plt.xlabel('Time (s)')
plt.ylabel('Beats Per Minute (BPM)')
plt.title('Heart Rate Data for Selected Workouts')
plt.show()

#Look into algorithm to reduce noise (or smooth) signal  