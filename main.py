import pandas as pd
import os
import matplotlib.pyplot as plt

from Data_plotter import Data_plotter

# Import Data
data_path = os.getcwd()+'\\Data'

Alcohol_sample_data     = pd.read_csv(os.path.join(data_path,'Alcohol_sample.csv'))
Non_Alcohol_sample_data = pd.read_csv(os.path.join(data_path,'no_Alcohol_sample.csv'))

# Analyze data
Data_prep = Data_plotter(Alcohol_sample_data, Non_Alcohol_sample_data)

# Plot descriptive statistic by driver for steering's signal
Data_prep.Plot_Steering_Statistics()

# Example for speed and acceleration signals
Data_prep.Plot_Speed_Acceleration()

# Add to dataFrame sub signal column
DF_with_sub_signals = Data_prep.get_DF_with_sub_signals()

# Visualize steering wheel acceleration signal in frequency domain
Data_prep.Plot_fft_visual()

# Plot steering wheel acceleration signal's insights (features)
Data_prep.Plot_Acc_FFT_Statistics()

# Plot peak detection to random driver's steering wheel signal
Data_prep.Plot_peak_detection()
plt.show()


