import numpy as np
import pandas as pd


class TEDADetect:
    """Class used to detect outliers on the dataset"""
    # ------------------------------
    # CONSTRUCTOR
    #-------------------------------
    def __init__(self, threshold=2):
        """
        Initializes a new TEDADetect object.

        Parameters:
            threshold (float): The threshold for outlier detection (default is 2).
        """
        try:
            # initialize variables
            self.k = 1
            self.variance = 0
            self.mean = 0
            self.threshold = threshold
        except Exception as e:
            print(f"Error during initialization: {e}")

    # ------------------------------
    # INTERNAL METHODS
    #------------------------------- 
    def __calcMean(self, x):
        """
        Calculates the updated mean.

        Parameters:
            x (float or numpy.ndarray): The new value or array.

        Returns:
            float or numpy.ndarray: The updated mean.
        """
        try:
            return ((self.k-1)/self.k)*self.mean + (1/self.k)*x
        except ZeroDivisionError:
            print("Error: Division by zero in __calcMean.")
    
    def __calcVariance(self, x):
        """
        Calculates the updated variance.

        Parameters:
            x (float or numpy.ndarray): The new value or array.

        Returns:
            float or numpy.ndarray: The updated variance.
        """
        try:
            distance_squared = np.square(np.linalg.norm(x - self.mean))
            return ((self.k-1)/self.k)*self.variance + distance_squared*(1/(self.k - 1))
        except Exception as e:
            print(f"Error in __calcVariance: {e}")
                                     
    def __calcEccentricity(self, x):
        """
        Calculates the eccentricity.

        Parameters:
            x (float or numpy.ndarray): The value or array.

        Returns:
            float or numpy.ndarray: The eccentricity.
        """
        try:
            if (isinstance(x, float)):
                if self.k != 0 and self.variance != 0:
                    return (1 / self.k) +  ((self.mean - x)**2 / (self.k *  self.variance))
                else:
                    return float('inf')     
            else:
                return (1 / self.k) +  (((self.mean - x).T.dot((self.mean - x))) / (self.k *  self.variance))
        except Exception as e:
            print(f"Error in __calcEccentricity: {e}")
            
        
    
    # ------------------------------
    # RUN METHODS
    #-------------------------------
    def run_offline(self, df, features):
        """
        Run the algorithm offline on a Dataframe.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the dataset.
            features (list): The list of feature column names.

        """
        try:
            # add is_outlier column to the dataframe
            df['is_outlier'] = 0
            
            # loop through the rows in df
            for index, row in df.iterrows():
                # build the X sample numpy array
                x = np.array(row[features])
                
                # update the model metrics
                if(self.k == 1):
                    self.mean = x
                    self.variance = 0
                else:
                    # calculate the new mean
                    self.mean = self.__calcMean(x)
                    # calculate the new variance
                    self.variance = self.__calcVariance(x)
                    # calculate the eccentricity and normalized eccentricity
                    eccentricity = self.__calcEccentricity(x)
                    norm_eccentricity = eccentricity/2
                    # define the threshold for outlier detection
                    threshold_ = (self.threshold**2 +1)/(2*self.k)
                    
                    # check if the point is an outlier
                    isOutlier = norm_eccentricity > threshold_

                    # if the point is an outlier, add it to the outlier list
                    if (isOutlier):
                        df.at[index, 'is_outlier'] = 1

                # Update the timestamp
                self.k = self.k + 1
        except Exception as e:
            print(f"Error in run_offline: {e}")

    def run_online(self, x):
        """
        Run the online algorithm on a single data point.

        Parameters:
            x (float or numpy.ndarray): The new data point.

        Returns:
            bool: True if the data point is an outlier, False otherwise.
        """
    
        try:
            is_outlier = False
            
            # update the model metrics
            if(self.k == 1):
                self.mean = x
                self.variance = 0
            else:
                # calculate the new mean
                self.mean = self.__calcMean(x)
                # calculate the new variance
                self.variance = self.__calcVariance(x)
                # calculate the eccentricity and nomalized eccentricity
                eccentricity = self.__calcEccentricity(x)
                norm_eccentricity = eccentricity/2
                # define the threshold for outlier detection
                threshold_ = (self.threshold**2 +1)/(2*self.k)
                
                # check if the point is an outlier
                is_outlier = norm_eccentricity > threshold_

            # Update the timestamp
            self.k = self.k + 1

            return is_outlier
        except Exception as e:
                print(f"Error in run_online: {e}")