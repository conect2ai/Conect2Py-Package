import numpy as np
import pandas as pd


class AutoTAC:    
    # ------------------------------
    # CONSTRUCTOR
    #-------------------------------
    def __init__(self, window_size=None, m=None, save_metrics=False):
        # initialize constructor params
        self.window_size = window_size
        self.m = 1 if m == None else m
        self.auto_adjust_m = True if m == None else False
        self.save_metrics = save_metrics
        # initialize internal variables
        self.time = 0
        self.k = 0
        self.variance = 0
        self.mean = 0
        self.window_count = 0
        
        self.last_value = None
        
        self.eccentricity = 0
        self.norm_eccentricity = 0
        self.outlier_limit = 0
        self.is_outlier = False
        self.should_keep = None
        
        self.fourth_moment = 0
        self.kurtosis = 0
   
        if(self.save_metrics): 
            self.metrics = []

    def __getCompressorInfo(self):
        return {
            'm': self.m,
            'window_size': self.window_size
        }
            

    # ------------------------------
    # INTERNAL METHODS
    #------------------------------- 
    def __saveMetrics(self, x):
        self.metrics.append({
            'time': self.time,
            'k': self.k,
            'mean': self.mean,
            'variance': self.variance,
            'eccentricity': self.eccentricity,
            'norm_eccentricity': self.norm_eccentricity,
            'outlier_limit': self.outlier_limit,
            'is_outlier': self.is_outlier,
            'should_keep': self.should_keep,
            'window_count': self.window_count,
            'fourth_moment': self.fourth_moment,
            'kurtosis': self.kurtosis,
            'm': self.m,
            'x': x 
        })
        
    
    def __resetWindow(self, x): 
        self.k = 0
        self.variance = 0
        self.mean = x
        self.window_count = 0
        self.last_value = x
        
        self.eccentricity = 0
        self.norm_eccentricity = 0
        self.outlier_limit = 0
        self.is_outlier = False
        
        self.fourth_moment = 0
        self.kurtosis = 0
        
        
    def __calcMean(self, x):
        new_mean = ((self.k-1)/self.k)*self.mean + (1/self.k)*x
        return new_mean
    
    
    def __calcVariance(self, x):
        distance_squared = (x - self.mean)**2
        new_var = ((self.k-1)/self.k)*self.variance + distance_squared*(1/(self.k-1))
        return new_var


    def __calcEccentricity(self, x):
        ecc = (1 / self.k) +  ((self.mean - x)**2 / (self.k * self.variance))
        return ecc


    def __calcOutlierLimit(self, x):
        limit = (self.m**2 + 1)/(2*self.k)
        return limit
    
    # ------------------------------
    # RUN METHODS
    #-------------------------------
    def check_point(self, x):
        """Evaluate a point"""

        # increment the total time and k
        self.time = self.time + 1
        self.k = self.k + 1
        should_reset_window = False
        
        # TEDA COMPUTATION
        
        if(self.k == 1):
            # update the model metrics
            self.mean = x
            self.variance = 0
            self.is_outlier = False
            
            # if its the first point in the whole series, save it
            if (self.time == 1):
                self.should_keep = True
            else:
                self.should_keep = False
        elif(x == self.last_value and self.variance == 0):
            # update the mean (is it needed?)
            self.mean = self.__calcMean(x)
            self.variance = self.__calcVariance(x)
            self.is_outlier = False
            self.should_keep = False
        else:
            # calculte the new mean, variance
            self.mean = self.__calcMean(x)
            self.variance = self.__calcVariance(x)
            # calculate the new eccentricity and normalized eccentricity
            self.eccentricity = self.__calcEccentricity(x)
            self.norm_eccentricity = self.eccentricity/2
            
            # TODO TEST Auto M
            # calculate the fourth moment and kurtosis
            self.fourth_moment = ((self.k-1)/self.k)*self.fourth_moment + (1/self.k)*((self.mean-x)**4)
            self.kurtosis = self.fourth_moment / self.variance**2
            # if (self.auto_adjust_m):
            #     if (-0.1 <= self.kurtosis <= 0.1):
            #         # distribution not prone to outliers
            #         self.m = 1
            #     elif (self.kurtosis > 0.1) and (self.kurtosis < 1.5):
            #         self.m = self.kurtosis
            #     else:
            #         self.m = 1/self.kurtosis
            if (self.auto_adjust_m):
                if (np.abs(self.kurtosis) <= 1):
                    self.m = np.abs(self.kurtosis)
                else:
                    self.m = np.abs(1/self.kurtosis)
            # TODO END TEST

            # define the threshold for outlier detection
            self.outlier_limit = self.__calcOutlierLimit(x)
            # check if the point is an outlier
            self.is_outlier = self.norm_eccentricity > self.outlier_limit

            # if the point is an outlier, increment the window_count
            if (self.is_outlier):
                self.window_count += 1

                if (self.window_size is None):
                    # if the window_size is none, always add the outlier
                     self.should_keep = True
                elif (self.window_count >= self.window_size):
                    # if the number of points in the window reaches the threshold
                    # reset the window and keep the point
                    should_reset_window = True
                    self.should_keep = True
                else: 
                    # discard the point
                    self.should_keep = False
            else:
                # discard the point
                self.should_keep = False
                
        # END OF TEDA COMPUTATION
            
        # update the metric history and return
        if(self.save_metrics):
            self.__saveMetrics(x)
            
        if(should_reset_window):
            self.__resetWindow(x)
            
        # update the last value
        self.last_value = x
        return self.should_keep