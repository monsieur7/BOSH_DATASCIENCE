from csv import QUOTE_NONE
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from scipy import stats
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
#TODO : readme and comments
#PARAMETERS :
average_summer = "19.2" #average temp in summer in germany
threshold = 2
threshold_components = 0.95
#END PARAMETERS
#https://github.com/andrewm4894/colabs/blob/master/time_series_anomaly_detection_with_pca.ipynb

def get_anomaly_scores(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=df_original.index)
    return loss
def is_anomaly(data, pca, threshold):
    pca_data = pca.transform(data)
    restored_data = pca.inverse_transform(pca_data)
    loss = np.sum((data - restored_data) ** 2)
    print(loss)
    return loss >= threshold
print(pd.__version__)

def dateParser(date):
    date = date.split('.')[0]
    return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")

season_find = "DSP_KK_CIRCUIT__24A1_TEMP_EXTERIEURE_EQUIP_ID_Value"
f = open("Data-Bosch/Data-Bosch/Daten_pandas.CSV", "r")
dataset = pd.read_csv(f, date_parser=dateParser, keep_default_na=False, quoting=QUOTE_NONE, engine="python")#, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])#, index_col=14, ) #delimiter=";", decimal="," maybe ?
date = dataset["timestamp"].to_numpy()
dataset = dataset.drop(dataset.columns[[dataset.shape[1]-2,dataset.shape[1]-1]], axis=1) #dropping some columns  that i dont want

#dataset = dataset.drop(dataset.columns[[i for i in range(0, dataset.shape[1], 2)]], axis=1)

print(dataset.columns)

week=1
array_season = dataset[season_find].to_numpy()
begin_summer = ""
summer=False
for i in range(0,len(array_season), 96*7): 
    avg = np.mean(array_season[i:i+96*7])# mean of a week 
    week+=1
    if(avg > 19.6 and summer==False): #  we are in summer if the average temp in a week is more than the threshold
        print("we are in summer", week, date[i])
        begin_summer=i
        summer=True

#split in two the arrays
dataset_winter = dataset.iloc[:begin_summer]
dataset_summer = dataset.iloc[begin_summer:]

merge = []

label_ct = 0
for data_season in (dataset_winter.to_numpy(), dataset_summer.to_numpy()):
    #scaling the data : centering it and reducing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_season)
    scaled_df = pd.DataFrame(scaled_data)

    pca = PCA(random_state=0)

    pca.fit_transform(scaled_data)
    #doing the pca 
    """
    if(label_ct == 0):
        plt.plot(pca.explained_variance_ratio_, label="explained variance ratio winter")

    else:
        plt.plot(pca.explained_variance_ratio_, label="explained variance ratio summer")
    """

    print(pca.explained_variance_ratio_)

    print("sum ", np.sum(pca.explained_variance_ratio_))
    sum = 0
    for ct, i in enumerate(pca.explained_variance_ratio_) :
        sum += i
        if(sum > threshold_components): # find how much components we want to keep
            break

    print("keeping", ct, "components")
    pca.set_params(n_components=ct) #keeping n components 
    pca.fit_transform(scaled_data) #must reload the data I know
    print("sum 2", np.sum(pca.explained_variance_ratio_))


    df_pca = pd.DataFrame(pca.fit_transform(scaled_data)) #put pca in a dataframe

    sns.lmplot(x="0", y="1", data=df_pca.rename(columns=lambda x: str(x)), fit_reg=True) #plot for debugging
    if(label_ct == 0): #title for the plot
        plt.title("winter")
    else:
        plt.title("summer")

    #plt.figure(200+i)
    #plt.scatter(df_pca[0], df_pca[1])
    #plt.show()

    df_restored = pd.DataFrame(pca.inverse_transform(df_pca)) # restoring the data
    df_restored = scaler.inverse_transform(df_restored) # de-scaling it 
    df_restored = pd.DataFrame(df_restored, columns=dataset.columns)

    merge.append(df_restored)
    label_ct += 1


"""
plt.title("explained variance ratio")
plt.legend(loc="upper left")
"""

df_restored = pd.concat(merge, ignore_index=True) #merging the two arrays
array = dataset.to_numpy()
num_errors = 0

i = 0
k = 0
j = 0 #TODO : add timestamp as a 4th column

new_dataset = dataset.copy()
print("old ", new_dataset.columns)
for _ in range(0, dataset.shape[1]): # adding the columns for the right data format
    new_dataset.insert(j, dataset.columns[i]+"HEADER", [dataset.columns[i] for _ in range(0, dataset.shape[0])])
    new_dataset.insert(j+2, dataset.columns[i]+"_FLAG", ["" for _ in range(dataset.shape[0])])
    new_dataset.insert(j+3, "TIMESTAMP", date, True) # adding timestamp

    i+=1
    j += 4
new_array = new_dataset.to_numpy()
print("new_dataset", new_dataset.columns)
print("value : ", new_dataset.to_numpy()[0,:])

for i in range(0, len(array[0])): # for each column
    plt.figure(i+20) # for the plot (new figure each time)

    num_errors = 0

    #threshold for winter / summer : mean +/- 2 sigma
    threshold_error_winter = (df_restored.to_numpy()[:,i] - array[:,i])**2
    threshold_i_winter = np.mean(threshold_error_winter) + threshold*np.std(threshold_error_winter)
    threshold_i_2_winter = np.mean(threshold_error_winter) - threshold*np.std(threshold_error_winter)

    threshold_error_summer = (df_restored.to_numpy()[:,i] - array[:,i])**2
    threshold_i_summer = np.mean(threshold_error_summer) + threshold*np.std(threshold_error_summer)
    threshold_i_2_summer = np.mean(threshold_error_summer) - threshold*np.std(threshold_error_summer)

    print("threshold", i, threshold_error_winter, threshold_error_summer)

    plt.plot(array[:,i])
    for j in range(0, len(array)):
        if(i < begin_summer ): #winter
            if(threshold_error_winter[j] > threshold_i_winter or threshold_error_winter[j] < threshold_i_2_winter): # it is an error
                new_array[j,k+2] = "TRUE" #not an error
                num_errors += 1

                plt.axvline(j, color='r', alpha=0.1) # putting a red line on the graphs
            else:
                new_array[j,k+2] = "FALSE" #not an error

    else: # summer
        if(threshold_error_summer[j] > threshold_i_summer or threshold_error_summer[j] < threshold_i_2_summer):
            new_array[j,k+2] = "TRUE"
            num_errors += 1

            plt.axvline(j, color='r', alpha=0.1)
        else:
            new_array[j,k+2] = "FALSE"

    k+=4 # iterating for each block of data
    plt.title(dataset.columns[i])
    print("name ", dataset.columns)
    print("number of errors detected", num_errors) 
    name=datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S") # saving the png files for each column
    name=str(name)+"_"+str(i)+"_" + ".png"
    plt.savefig(name)

name_file = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S_FLAG.csv")
print("shape ", new_array.shape)
new_dataset = pd.DataFrame(new_array, columns=new_dataset.columns) #putting it in a new dataframe.
new_dataset.to_csv(name_file, index=False) #exporting it in a csv file
plt.show() # show the graphs
