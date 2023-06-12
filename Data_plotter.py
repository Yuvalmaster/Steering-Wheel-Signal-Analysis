import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from Driver import Driver
from Subject_Analysis import Subject_Analysis

class Data_plotter:
    def __init__(self, Alcoholic_data: pd.DataFrame, Non_Alcoholic_data: pd.DataFrame):
       
        self.subjects          = []                                                 # Create list of subjects' objects
        self.subjects_analysis = []                                                 # Create list of subjects' analyses objects
             
        concat_data    = pd.concat([Alcoholic_data, Non_Alcoholic_data],axis=0)    
        IDs, IDs_order = np.unique(concat_data.driver, return_index=True)
        self.IDs       = IDs[np.argsort(IDs_order)]                                 # Gets the IDs sorted by location in concat data
        
        # Extract data for each subject (driver)
        for ind, ID in enumerate(self.IDs):
            driver_data = concat_data.query('driver == @ID')
            self.subjects.append(Driver(ID, driver_data)) 
            self.subjects_analysis.append(Subject_Analysis(self.subjects[ind].Steering, 
                                                           self.subjects[ind].TimeStamp))
   
    # %% PLOTS   
    def Plot_Steering_Statistics(self):
        # This function plots the descriptive statistics of drivers' steering wheel signals
        num_of_stats       = len(self.subjects_analysis[0].steering_features)           # Number of statistics to show
        keys_stats         = list(self.subjects_analysis[0].steering_features.keys())   # description of each stat (feature)
        drivers_stats      = np.zeros((len(self.subjects),num_of_stats))                # preallocation with zeros array
        drivers_alcohol_cc = []
        rows               = 2
        cols               = int(np.ceil(num_of_stats/rows))
        
        # Creating a matrix of features per subject (Rows = subjects, Columns = features)
        for ind, driver in enumerate(self.subjects):
            drivers_stats[ind,:] = np.array(list(self.subjects_analysis[ind].steering_features.values()))
            drivers_alcohol_cc.append(driver.Alcohol_cc) 
            
        labels  = [str(i) for i in drivers_alcohol_cc]
        
        fig0, ax = plt.subplots(rows, cols, figsize=(40,20))
        ax      = ax.flatten()
        # Plot in barplot style - X axis: alcohol consumed (per subject), Y axis: feature value (per subject)
        for i, key in enumerate(keys_stats):
            
            ax[i].bar(range(len(labels)), drivers_stats[:,i],  0.5); 
            ax[i].set_xticks(range(len(labels))); ax[i].set_xticklabels(labels)
            ax[i].set_xlabel('Alcohol consumed [cc]'); ax[i].set_ylabel(key); ax[i].set_title(key + '\nSteering wheel per driver')
            
            # Add driver name to each column
            for j,name in enumerate(self.IDs):
                ax[i].text(j, drivers_stats[j,i], name, ha='center')
        
        fig0.suptitle("Time Domain: Steering's signal features")
        fig0.show()  
    
    def Plot_Speed_Acceleration(self):
        # This function plots speed and normalized acceleration of random driver's steering wheel signal. 
        chosen_subject = random.randint(0, len(self.subjects)-1)
        subject        = self.subjects_analysis[chosen_subject]
        
        fig1, ax = plt.subplots(2,1,figsize=(15,10))
        ax       = ax.flatten()
        
        ax[0].plot(subject.TimeStamp[0:subject.speed.size], subject.speed)
        ax[0].set_xlabel('TimeStamp [sec]'); ax[0].set_ylabel('Steering Speed [degree°/sec]')
        ax[0].set_title('Steering wheel speed:\n' + self.subjects[chosen_subject].ID)
    
        ax[1].plot(subject.TimeStamp[0:subject.acc_normalized.size], subject.acc_normalized)
        ax[1].set_xlabel('TimeStamp [sec]'); ax[1].set_ylabel('Normalized Steering Acceleration [degree°/sec^2]')
        ax[1].set_title('Steering wheel Acceleration - Normalized:\n' + self.subjects[chosen_subject].ID)
        fig1.show()
    
    def Plot_fft_visual(self):
        # This function plot a visualization of steering wheel acceleration signal in frequency domain for two drivers,
        # one not consumed alcohol, and one that does. In addition, each graph has a fitted baseline. 
        chosen_drivers       = [self.subjects[0], self.subjects[-1]]
        chosen_drivers_stats = [self.subjects_analysis[0], self.subjects_analysis[-1]]
        
        fig2, ax = plt.subplots(2,1,figsize=(15,10))
        ax      = ax.flatten()
        
        for i, driver in enumerate(chosen_drivers_stats):
            ax[i].plot(driver.freqs, driver.fft_acc,  label='FFT')
            ax[i].plot(driver.freqs, driver.baseline, label='Baseline')
            
            ax[i].legend()
            ax[i].set_xlabel('[Hz]'); ax[i].set_ylabel('dB'); ax[i].set_title(chosen_drivers[i].ID + '\nSteering wheel acceleration: Frequency Domain')
        
        fig2.suptitle("Frequency Domain: Steering's acceleration signal")
        fig2.show()  
        
   
    def Plot_Acc_FFT_Statistics(self):
        # This function plots the insights of drivers' steering wheel acceleration signals in frequency domain
        num_of_stats       = len(self.subjects_analysis[0].steering_acc_features)           # Number of statistics to show
        keys_stats         = list(self.subjects_analysis[0].steering_acc_features.keys())   # description of each stat (feature)
        drivers_stats      = np.zeros((len(self.subjects),num_of_stats))                    # preallocation with zeros array
        drivers_alcohol_cc = []
        rows               = 2
        cols               = int(np.ceil(num_of_stats/rows))
        
        # Creating a matrix of features per subject (Rows = subjects, Columns = features)
        for ind, driver in enumerate(self.subjects):
            drivers_stats[ind,:] = np.array(list(self.subjects_analysis[ind].steering_acc_features.values()))
            drivers_alcohol_cc.append(driver.Alcohol_cc) 
            
        labels  = [str(i) for i in drivers_alcohol_cc]
        
        fig3, ax = plt.subplots(rows, cols, figsize=(40,20))
        ax      = ax.flatten()
        # Plot in barplot style - X axis: alcohol consumed (per subject), Y axis: feature value (per subject)
        for i, key in enumerate(keys_stats):
            
            ax[i].bar(range(len(labels)), drivers_stats[:,i],  0.5); 
            ax[i].set_xticks(range(len(labels))); ax[i].set_xticklabels(labels)
            ax[i].set_xlabel('Alcohol consumed [cc]'); ax[i].set_ylabel(key); ax[i].set_title(key + '\nSteering wheel acceleration: Frequency Domain')
            
            # Add driver name to each column
            for j,name in enumerate(self.IDs):
                ax[i].text(j, drivers_stats[j,i], name, ha='center')
        
        fig3.suptitle("Frequency Domain: Steering wheel acceleration's signal features")
        fig3.show() 
        
   
    def Plot_peak_detection(self):
        # This function plots random driver's steering wheel signal with peak detection in overlay.
        chosen_subject = random.randint(0, len(self.subjects)-1)
        subject        = self.subjects_analysis[chosen_subject]
        
        fig4 = plt.figure()
        plt.plot(subject.TimeStamp, subject.Steering)
        plt.scatter(subject.TimeStamp[subject.peaks_locs], subject.Steering[subject.peaks_locs], color='red')
        plt.xlabel('TimeStamp [sec]'); plt.ylabel('Steering [degree°]')
        plt.title('Peak Detection algorithm visualization:\n' + self.subjects[chosen_subject].ID)
        
        fig4.show() 
                 
    
    # %% DATASETS
    def get_DF_with_sub_signals(self):
        # This function creates a csv file and a pandas DataFrame with sub signals label to each entry for each subject, with data manipulation
        # to the TimeStamp and Steering.
        DF_with_sub_signals = pd.DataFrame()
        
        for _,subject in enumerate(self.subjects):    
            df                  = subject.create_personal_DF()
            DF_with_sub_signals = pd.concat([DF_with_sub_signals, df], axis=0)
        
        DF_with_sub_signals.to_csv('Dataset_with_sub_signals.csv', encoding='utf-8')
        return DF_with_sub_signals



            
