#!/usr/bin/env python
# coding: utf-8

# # Data Analysis on Airlines Delays

# #### Goal :                                                                                                  
# To analyse airlines delay data in USA and curve delays caused in major airlines between busiest routes which will give a fair idea to passengers about frequency of delays in general from last two years along with a particular airline and between a particular route. The same will also help airlines for better management and work on root cause of the delays.

# #### Objective: 
# To provide general descriptive analysis on delays from last two years 
# To analyse the root cause of delays
# To compare between different airlines and major airports with respect to delays and cancellation
# Scope of Predictive analysis in determining probabale delays in defined airlines and airports

# In[267]:


import unicodecsv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy import stats
from scipy.stats import norm


# ### Data Cleaning of the dataset:

# In[268]:


# Read csv file into a pandas dataframe.
Departure = pd.read_csv(r"C:/Users/varap/OneDrive/Documents/Intro to DS/Project/Consolidated Raw Files/Files/Files_3rdNov_1113PM/Departure.csv", encoding = 'unicode_escape')


# In[269]:


# Read csv file into a pandas dataframe.
Arrivals = pd.read_csv(r"C:/Users/varap/OneDrive/Documents/Intro to DS/Project/Consolidated Raw Files/Files/Files_3rdNov_1113PM/Arrival.csv", encoding = 'unicode_escape')


# In[270]:


Arrivals.columns


# In[271]:


Arrivals.shape


# In[272]:


# Read csv file into a pandas dataframe.
Cancellation = pd.read_csv(r"C:/Users/varap/OneDrive/Documents/Intro to DS/Project/Consolidated Raw Files/Files/Files_3rdNov_1113PM/Cancellation.csv", encoding = 'unicode_escape')


# In[273]:


Cancellation.shape


# In[274]:


class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


# In[275]:


df = pd.merge(Departure,Arrivals, on=['Carrier Code', 'Date (MM/DD/YYYY)','Flight Number','Tail Number'],how='outer')


# In[276]:


df2 = pd.merge(df,Cancellation,on=['Carrier Code', 'Date (MM/DD/YYYY)','Flight Number'],how='outer')


# In[277]:


df2.fillna({'Cancellation Code':0},inplace=True)


# In[278]:


df3 = df2.drop(["Scheduled departure time","Actual departure time","Wheels-off time","Origin Airport_y","Scheduled Arrival Time","Actual Arrival Time","Wheels-on Time","Wheels-off time","Destination Airport_y","Delay Carrier (Minutes)_y","Delay Weather (Minutes)_y","Delay National Aviation System (Minutes)_y","Delay Security (Minutes)_y", "Delay Late Aircraft Arrival (Minutes)_y","Destination Airport_y", "Tail Number_y", "Destination Airport","Origin Airport"],axis=1)


# In[279]:


df3.columns


# In[280]:


df3.shape


# In[281]:


df4 = df3.dropna()


# In[282]:


df4.shape


# In[283]:


df4.isnull().sum()


# In[284]:


df4.columns


# In[316]:


df4.rename(columns={'Carrier Code':'Carrier_Code','Date (MM/DD/YYYY)':'Date','Flight Number':'Flight_Number','Tail Number_x':'Tail_Number','Origin Airport_x':'Origin_Airport','Destination Airport_x':'Destination_Airport','Scheduled elapsed time (Minutes)':'Scheduled_elapsed_time','Actual elapsed time (Minutes)':'Actual_elapsed_time','Departure delay (Minutes)':'Departure_delay','Taxi-Out time (Minutes)':'Taxi-Out_Time','Delay Carrier (Minutes)_x':'Carrier_Delay','Delay Weather (Minutes)_x':'Weather_Delay','Delay National Aviation System (Minutes)_x':'National_Aviation_System_Delay','Delay Security (Minutes)_x':'Security_Delay','Delay Late Aircraft Arrival (Minutes)_x':'Late_Aircraft_Arrival_Delay','Arrival Delay (Minutes)':'Arrival_Delay', 'Taxi-In time (Minutes)':'Taxi-In_Time','Cancellation Code':'Cancellation_Code'},inplace=True)


# In[286]:


df4.columns


# In[317]:


# Formatting Date Column to 'YYYY-MM-DD' format:
df4['Date']=pd.to_datetime(df4.Date,format='%m/%d/%Y')
df4.head()


# In[288]:


# Sorting the data by Date Column in Ascending order:
df4=df4.sort_values(by=['Date'],ascending=True)
df4[0:10]


# In[289]:


# Checking for duplicate Values
df4.duplicated().sum()


# In[290]:


# dropping duplicate values
df4.drop_duplicates(keep=False,inplace=True)


# In[291]:


df4.duplicated().sum()


# In[292]:


df4.columns


# In[293]:


df4['Total_Delay'] = df4['Departure_delay'] + df4['Arrival_Delay']


# In[294]:


df4.head()


# In[295]:


# Status represents wether the flight was on time (0), slightly delayed (1), highly delayed (2), diverted (3), or cancelled (4)


# In[296]:


df4.loc[df4['Total_Delay'] <= 15, 'Status'] = 0
df4.loc[df4['Total_Delay'] >= 15, 'Status'] = 1
df4.loc[df4['Total_Delay'] >= 60, 'Status'] = 2
df4.loc[df4['Cancellation_Code'] == 1, 'Status'] = 3


# In[297]:


df4.head()


# In[298]:


# Creating new columns month and year using the 'Date' Column:
df4['year'] = df4['Date'].dt.year
df4['month'] = df4['Date'].dt.month


# In[299]:


df4.tail()


# ### Exploratory Data Analysis:

# In[301]:


f,ax=plt.subplots(1,2,figsize=(20,8))
df4['Status'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Status')
ax[0].set_ylabel('')
sns.countplot('Status',order = df4['Status'].value_counts().index, data=df4,ax=ax[1])
ax[1].set_title('Status')
plt.show()

print('Status represents whether the flight was on time (0), slightly delayed (1), highly delayed (2), or cancelled (3)')


# In[302]:


Delayedflights = df4[(df4.Status >= 1) &(df4.Status < 3)]
sns.distplot(Delayedflights['Total_Delay'])
plt.xlabel('Delay in mins')
plt.ylabel('Density')
plt.title('Density Plot and Histogram of Flight Delays')
plt.show()


# In[303]:


print("Skewness: %f" % Delayedflights['Arrival_Delay'].skew())
print("Kurtosis: %f" % Delayedflights['Arrival_Delay'].kurt())


# In[304]:


# It can be seen on the histogram and by the skewness and kurtosis indexes, that delays are mostly located on the left side of the graph, with a long tail to the right. The majority of delays are short, and the longer delays, while unusual, are more heavy loaded in time.


# In[305]:


flights_by_carrier = df4.pivot_table(index='Date', columns='Carrier_Code', values='Flight_Number', aggfunc='count')
flights_by_carrier.head()


# In[306]:


df4.pivot_table(columns='Date')


# In[307]:


delays_list = ['Carrier_Delay','Weather_Delay','Late_Aircraft_Arrival_Delay','National_Aviation_System_Delay','Security_Delay']
flight_delays_by_day = df4.pivot_table(index='Date', values=delays_list, aggfunc='sum')


# In[308]:


flight_delays_by_day.plot(kind='area', figsize=[16,6], stacked=True, colormap='autumn') # area plot
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Relative Distribution of Delays')
plt.legend()
plt.show()


# In[309]:


df4['Carrier_Code'].value_counts()[:20].plot(kind='barh')
plt.xlabel('count of delays')
plt.ylabel('Airlines')
plt.title('Delays v/s Airlines')
plt.show()


# In[310]:


f,ax=plt.subplots(1,2,figsize=(20,8))
Delayedflights[['month','Total_Delay']].groupby(['month']).mean().plot(ax=ax[0])
ax[0].set_title('Average delay by month')
Delayedflights[['month','Total_Delay']].groupby(['month']).sum().plot(ax=ax[1])
ax[1].set_title('Number of minutes delayed by month')
plt.show()


# In[311]:


df4 = Delayedflights.filter(['month','Carrier_Delay','Weather_Delay','National_Aviation_System_Delay','Security_Delay','Late_Aircraft_Arrival_Delay'], axis=1)
df4 = df4.groupby('month')['Late_Aircraft_Arrival_Delay','Carrier_Delay','Weather_Delay','National_Aviation_System_Delay','Security_Delay'].sum().plot()
df4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
plt.xlabel('Month')
plt.ylabel('Delays')
plt.show()


# In[312]:


f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('Carrier_Code','Carrier_Delay', data=Delayedflights,ax=ax[0])
ax[0].set_title('Average Delay by Carrier')
sns.boxplot('Carrier_Code','Carrier_Delay', data=Delayedflights,ax=ax[1])
ax[1].set_title('Delay Distribution by Carrier')
plt.close(2)
plt.show()
print(['AA: American Airlines','UA: United Airlines','DL: Delta Airlines','B6: JetBlue Airways','AS: Alaska Airlines'])


# In[313]:


ListOfAirports = Delayedflights[(Delayedflights.Origin_Airport == 'ATL') | (Delayedflights.Origin_Airport == 'LAX') | (Delayedflights.Origin_Airport == 'SFO')]
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('Origin_Airport','National_Aviation_System_Delay', data=ListOfAirports,ax=ax[0])
ax[0].set_title('Average Delay by Origin Airport')
sns.boxplot('Origin_Airport','National_Aviation_System_Delay', data=ListOfAirports,ax=ax[1])
ax[1].set_title('Delay Distribution by Origin Airport')
plt.close(2)
plt.show()


# In[315]:


cols = ['Arrival_Delay', 'Carrier_Delay', 'Late_Aircraft_Arrival_Delay', 'National_Aviation_System_Delay', 'Weather_Delay','Security_Delay']
sns.pairplot(Delayedflights[cols], size = 2.5)
plt.show()


# #### Target audience: 
# Airlines, Airport authority, Passengers
