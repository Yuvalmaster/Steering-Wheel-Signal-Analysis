import numpy as np
import scipy.stats as sp
from scipy.signal import medfilt, convolve
from scipy.integrate import cumtrapz

class Subject_Analysis:
    def __init__(self, Steering_signal: np.array, TimeStamp: np.array):
        # Importing the steering signal and TimeStamp for specific driver.
        self.Steering  = Steering_signal
        self.TimeStamp = TimeStamp

        # Calculating speed, acceleration and normalizing acceleration.
        self.speed          = self.get_steering_speed()
        self.acceleration   = self.get_steering_acceleration()
        self.acc_normalized = self.Normalize_signal(self.acceleration)
      
        # Calculate frequency domain steering acceleration signal.
        self.fft_acc, self.freqs = self.get_frequency_domain_acceleration()
        self.baseline            = self.baseline_fitting(self.fft_acc)
        
        # Extracting features for both steering signal and steering acceleration 
        self.steering_features     = self.get_steering_features()
        self.steering_acc_features = self.get_steering_acceleration_features()
        
        # Extract peaks locations
        self.peaks_locs = self.get_peaks_locs()

    # %% ATTRIBUTES FUNCTIONS
    def get_steering_speed(self) -> np.array:
        # This function calculate the steering speed
        speed = self.derivative(self.Steering, self.TimeStamp)
        return speed
    
    def get_steering_acceleration(self) -> np.array:
        # This function calculate the steering acceleration, based on the steering speed.
        if hasattr(self,'speed') == False:
            self.speed = self.get_steering_speed()
        acceleration = self.derivative(self.speed, self.TimeStamp[0:len(self.speed)])
        return acceleration
    
    def get_frequency_domain_acceleration(self):
        # This function get the frequency domain of the steering acceleration, and slightly filtering the signal.
        Fs             = self.sampling_frequency(self.TimeStamp)     # Get sampling frequency
        fft_acc, freqs = self.fft_signal(self.acc_normalized, Fs)    # fourier transform over normalized signal 
        fft_acc        = self.fft_filtering(fft_acc)                 # Filtering outliers and smoothing
        return fft_acc, freqs
    
    def get_peaks_locs(self):
        # This function find the peaks location of the Steering signal
        Fs               = self.sampling_frequency(self.TimeStamp)
        window_size      = int(np.floor(Fs)) # set the window size to approximately 1 [sec]
        threshold_factor = 0.15
        peak_locs        = self.Peaks_detector(self.Steering, threshold_factor, window_size)
        return peak_locs
    
    # %% PROCESSING
    # =========== Time Domain ========== #
    @staticmethod 
    def derivative(Steering_signal: np.array, TimeStamp: np.array) -> np.array:
        # Calculate derivative according to dy/dt
        dy   = Steering_signal[1:-1] - Steering_signal[0:-2]
        dt   = TimeStamp[1:-1] - TimeStamp[0:-2] 
        grad = dy/dt
        return grad 
    
    @staticmethod 
    def Normalize_signal(signal: np.array) -> np.array:
        # Calculate mean and standard deviation
        mean = np.mean(signal)
        std  = np.std(signal)
        
        # Normalize based on standard normal distribution
        normalized_signal = (signal - mean) / std
        return normalized_signal 
    
    # ========= Frequency Domain ========= #    
            
    @staticmethod 
    def sampling_frequency(TimeStamp: np.array) -> float:
        # Calculate the sampling rate based on average between timestamps
        Fs = 1/np.mean(np.diff(TimeStamp))
        return Fs
    
    @staticmethod 
    def fft_signal(signal: np.array, Fs: float):
        # This function transform the signal to frequency domain using fourier transform
        fft_sig = np.abs(np.fft.fft(signal))**2 / len(signal)   # Compute power spectrum
        fft_sig = 10 * np.log10(fft_sig)                        # Convert to decibels [dB]
        
        freqs   = np.fft.fftfreq(signal.size, 1/Fs)    
        
        # Ignore symmetry
        fft_sig = fft_sig[freqs >= 0]
        freqs   = freqs[freqs >= 0]
        
        return fft_sig, freqs
    
    @staticmethod
    def fft_filtering(signal: np.array, smooth=1) -> np.array:
        # This function filtering the fft signal with median filter and slightly smoothing using convolution.
        # The reason for the filtering is to denoise the signal from artifacts and to visualize the baseline.
        filtered_fft = medfilt(signal, kernel_size=3)  # using median filter with kernel size 3 to address outliers
        
        if smooth == 1:                                # if smoothing filter is ON (=1, OFF=0), the signal will be smoothed using moving average
            window_size   = 10
            filter_kernel = np.ones(window_size) / window_size
            filtered_fft  = convolve(filtered_fft, filter_kernel, mode='same') # using zero padding to maintain the same signal length  
        
        return filtered_fft
    
    @staticmethod
    def baseline_fitting(signal: np.array) -> np.array:
        # Fit a 5rd-degree polynomial to the signal
        polyfit  = np.polyfit(np.arange(len(signal)), signal, deg=5)
        baseline = np.polyval(polyfit, np.arange(len(signal))) 
        return baseline
      
    @staticmethod
    def db2PSD_converter(signal: np.array) -> np.array:
        PSD_sig = (10**(-signal/10))**2
        return PSD_sig

    @staticmethod
    def find_cut_off_freq(signal: np.array, freqs: np.array) -> float:
        # The cutoff frequency is the frequency at which the signal's response is attenuated by -3dB from its maximum frequency response.
        # This function finds the cutoff frequency around the maximum frequency response.
        max_loc     = np.argmax(signal)
        cut_off_val = np.max(signal) - 3 
        search_inds = np.where(signal <= cut_off_val)
        return freqs[np.argmin(abs(search_inds-max_loc))]
    
    @staticmethod
    def calculate_hist(signal: np.array) -> np.array:
        histogram,_ = np.histogram(signal, density=True) # get probability for each value in signal
        return histogram
   
    # %% FEATURES                
    def get_steering_features(self) -> dict:        
        Mean   = np.mean(self.Steering)
        Std    = np.std(self.Steering)
        Median = np.median(self.Steering)
        Max    = np.max(self.Steering)
        Min    = np.min(self.Steering)
         
        h_precentile = np.percentile(self.Steering, 75)
        l_precentile = np.percentile(self.Steering, 25)
 
        entropy  = sp.entropy(self.calculate_hist(self.Steering))
        skewness = sp.skew(self.Steering)
        rms      = np.sqrt(np.mean(self.Steering**2))
        
        number_of_turning = self.steering_turning_feature()
        stable_time       = self.stable_time()
                     
        features = {
            'Min [degree°]'             : Min,
            'Max [degree°]'             : Max,
            'Mean [degree°]'            : Mean,
            'std [degree°]'             : Std,
            'Median [degree°]'          : Median,
            '75th precentile [degree°]' : h_precentile,
            '25th precentile [degree°]' : l_precentile,
            'Entropy'                   : entropy,
            'Skewness'                  : skewness,
            'Root Mean Square'          : rms,
            'Number of turning'         : number_of_turning,
            'Stable time'               : stable_time           
            }
        
        return features
        
    def get_steering_acceleration_features(self) -> np.array:
        # Frequency domain features
        slow_indices = np.logical_and(self.freqs >= 1, self.freqs <= 25)
        fast_indices = np.logical_and(self.freqs >= 150, self.freqs <= 250)    
        sig_baseline = self.baseline_fitting(self.fft_acc)    
        
        max_frequency        = self.freqs[np.argmax(self.fft_acc)]                                      # The frequency of the maximum frequency response
        SlowFast_freqs_ratio = abs(sum(self.fft_acc[slow_indices]) / sum(self.fft_acc[fast_indices]))   # The ratio between the sum of low frequency responses and high frequency responses
        cutoff_freq          = self.find_cut_off_freq(sig_baseline, self.freqs)                         # The Cutoff frequency according to signal's baseline!
       
        # Power Spectral Density features
        PSD_sig = self.db2PSD_converter(self.fft_acc)
        cdf     = cumtrapz(PSD_sig, self.freqs)
        cdf    /= cdf[-1] # Normalize to 1 
        
        spectral_entropy  = sp.entropy(self.calculate_hist(PSD_sig))                                    # Entropy value of the signal's PSD
        spectral_centroid = np.sum(PSD_sig * self.freqs) / np.sum(PSD_sig)                              # Centroid value of the signal's PSD
        median_freq       = self.freqs[np.argmin(np.abs(cdf-0.5))]                                      # Median frequency based on signal's PSD
        
        features = {
            'Dominant frequency [Hz]'            : max_frequency,
            'Slow/Fast acceleration ratio'       : SlowFast_freqs_ratio,
            'Cutoff Frequency [Hz]'              : cutoff_freq,
            'Power Spectral Density Entropy'     : spectral_entropy,
            'Power Spectral Density Centroid'    : spectral_centroid,
            'Median Frequency [Hz]'              : median_freq,
            }
        
        return features
    
    ## UNIQUE FEATURES  
    def steering_turning_feature(self) -> int:
        # Calculate the number of turning the subject did.
        directions         = np.sign(self.Steering)
        direction_changing = np.diff(directions[directions != 0]) # find all transitions from positive to negative, ignoring transition from either to zero.
        number_of_turning  = sum(direction_changing != 0)         # count all the turning -> moving the steering wheel from left to right and vice versa.
        return number_of_turning
    
    def stable_time(self) -> float:
        # Calculate the relative stable steering wheel time during the entire session for a subject.
        steering_diff = np.concatenate((np.diff(self.Steering),
                                        np.diff(self.Steering[-2:]))) # Adding diff of last timestamps to maintain vector size.
        stable_time = self.TimeStamp[steering_diff == 0]
        return sum(stable_time) / self.TimeStamp[-1]                  # Normalize the stable time by total session time.
        

    # %% FIND PEAKS
    @staticmethod
    def Peaks_detector(signal: np.array, threshold_factor: float, window_size: int) -> np.array:
        # This is a basic algorithm for peak detection with adaptive threshold using moving windows.
        # The algorithm can be improved by further analyses of the data, hyperparameter optimization,
        # or by using derivatives to have more accurate results (such as AF2 peak detection algorithm).
        abs_sig = abs(signal)                                                   # Addressing both negative and positive peaks       
        
        local_max = np.zeros(signal.shape)                                      # Compute the local maximum in a sliding window    
        for i in range(window_size, signal.shape[0] - window_size):
            local_max[i] = np.max(abs_sig[i - window_size:i + window_size])  

        threshold = threshold_factor * np.max(local_max)                        # Compute local thresholds
        
        peaks = np.where((abs_sig > threshold) & (abs_sig == local_max))[0]     # Find the indices where the signal is above the threshold and is a local maximum
        peaks = peaks[np.where(np.diff([peaks]) > 1)[1]]                        # Remove adjacent indices
              
        return peaks            
        
            
        
        

        
        
    
    

        
    

            
