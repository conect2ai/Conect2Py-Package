import numpy as np
import pandas as pd

class TAC:
    """
    Class used to compress IoT sensors data stream
        window_threshold: number of outliers that need to be detected to generate a new window
        m: parameter used to define the outlier threshold
        save_metrics: if True, saves the metrics for each point in the model
    """
    
    # ------------------------------
    # CONSTRUCTOR
    #-------------------------------
    def __init__(self, m=1, window_threshold=1, save_metrics=False):
        # initialize constructor params
        self.m = m
        self.window_threshold = window_threshold
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
        self.outlier_threshold = 0
        self.is_outlier = False
        self.should_keep = None
        
        if(self.save_metrics): 
            self.metrics = []
            

    # ------------------------------
    # INTERNAL METHODS
    #------------------------------- 
    def __saveMetrics(self, x):
        self.metrics.append({
            'k': self.k,
            'mean': self.mean,
            'variance': self.variance,
            'eccentricity': self.eccentricity,
            'norm_eccentricity': self.norm_eccentricity,
            'outlier_limit': self.outlier_threshold,
            'is_outlier': self.is_outlier,
            'should_keep': self.should_keep,
            'window_count': self.window_count,
            'x': x,
        })
        
    
    def __resetWindow(self, x, time='False'): 
        self.k = 0
        self.variance = 0
        self.mean = x
        self.window_count = 0
        self.last_value = x
        if time:
            self.time = 0
        
        self.eccentricity = 0
        self.norm_eccentricity = 0
        self.outlier_threshold = 0
        self.is_outlier = False
        
        
    def __calcMean(self, x):
        new_mean = ((self.k-1)/self.k)*self.mean + (1/self.k)*x
        return new_mean
    
    def __calcVariance(self, x):
        distance_squared = (x - self.mean)**2
        new_var = ((self.k-1)/self.k)*self.variance + distance_squared*(1/(self.k-1))
        return new_var
                                     
    def __calcEccentricity(self, x):
        return (1 / self.k) +  ((self.mean - x)**2 / (self.k * self.variance))

    def check_point(self, x):
        """Evaluate a point"""
        # For 1D n is always 1
        # n = 0.8
        
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

            # define the threshold for outlier detection
            self.outlier_threshold = (self.m**2 +1)/(2*self.k)
            # check if the point is an outlier
            self.is_outlier = self.norm_eccentricity > self.outlier_threshold
            # if the point is an outlier, increment the window_count
            if (self.is_outlier):
                self.window_count += 1

            # if the number of points in the window reaches the threshold
            # reset the window and keep the point
            if(self.window_count >= self.window_threshold):
                # reset the window
                should_reset_window = True
                self.should_keep = True
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