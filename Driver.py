import numpy as np
import pandas as pd

class Driver:
    # Limits the number of attributes for driver.
    __slots__ = ['ID', 'NumTrial', 'TimeStamp', 'Steering', 'Acceleration', 'Brake', 'Alcohol_cc','sub_signal']  
    def __init__(self, ID: str, Driver_data: pd.DataFrame):
        
        # Data preprocessing
        Driver_data = Driver_data.drop_duplicates(subset=['TimeStamp'])  # Address time stamps duplicates (if exist).
        Driver_data = Driver_data.dropna(subset=['TimeStamp'])           # Address NA/Null values (if exist).   
        
        self.ID           = ID
        self.NumTrial     = Driver_data.NumTrial.to_numpy()
        
        timestamp         = Driver_data.TimeStamp.to_numpy()
        self.TimeStamp    = (timestamp - timestamp[0])/1000              # Normalize the time stamp to seconds, starting from zero for each driver.
        
        self.Steering     = Driver_data.Steering.to_numpy()*180          # Convert Steering wheel signal to angles.                                
        self.Acceleration = Driver_data.Acceleration.to_numpy()
        self.Brake        = Driver_data.Brake.to_numpy()
        self.Alcohol_cc   = Driver_data.Alcohol_cc.to_numpy()[0]         # the Alcohol cc is a constant, thus retrieving only the first index.
        self.sub_signal   = self.data_segmentation()
    
    def __repr__(self):
        if max(self.Alcohol_cc > 0):
            return "Driver ID: " + self.ID + '\n' +\
                   "Alcohol: Under influence"
        else:
            return "Driver ID: " + self.ID + '\n' +\
                   "Alcohol: Not under influence"    
                                
    def data_segmentation(self) -> np.array:
        sub_signal_inds = np.floor(self.TimeStamp)
        return sub_signal_inds
 
    def create_personal_DF(self) -> pd.DataFrame:
        # This function recreate new data frame based on the data manipulation of the class, with sub signal labels included.
        df = pd.DataFrame({
            'ID'                : self.ID,
            'NumTrial'          : self.NumTrial,
            'TimeStamp [sec]'   : self.TimeStamp,
            'Steering [degree°]': self.Steering,
            'Acceleration'      : self.Acceleration,
            'Brake'             : self.Brake,
            'Alcohol [cc]'      : self.Alcohol_cc,
            'Sub Signal '       : self.sub_signal
            })    
        return df
               

     