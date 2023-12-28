import numpy as np
import pandas as pd

class DataCloud:
    """Represents the datacloud objects"""
    # ------------------------------
    # CONSTRUCTOR
    # ------------------------------
    def __init__(self, x, point_key=None):
        """
        Initializes a new DataCloud object.

        Parameters:
            x (float): The initial value for mean.
            point_key (hashable, optional): The key for the point (default is None).
        """
        try:
            # initialize new datacloud stats with the point
            self.n = 1 
            self.points = {point_key} # set of point keys that belog to the cloud 
            self.mean = x
            self.var = 0
        except Exception as e:
            print(f"Error initializing datacloud: {e}")

    # ------------------------------
    # EXTERNAL METHODS
    # ------------------------------
    def get_updated_mean(self, x):
        """
        Calculates the updated mean.

        Parameters:
            x (float): The new value.

        Returns:
            float: The updated mean.
        """
        try:    
            new_n = self.n + 1
            return ((new_n - 1)/ new_n)*self.mean + (1/new_n)*x
        except ZeroDivisionError:
            print("Error: Division by zero in get_updated_mean.")
    
    def get_updated_variance_k_2(self, x):
        """
        Calculates the updated variance using method k=2.

        Parameters:
            x (float): The new value.

        Returns:
            float: The updated variance.
        """
        try:
            return np.linalg.norm((x - self.mean)/2)
        except Exception as e:
            print(f"Error in get_updated_variance_k_2: {e}")


    def get_updated_variance_k_3(self, x, test_mean):
        """
        Calculates the updated variance using method k=3.

        Parameters:
            x (float): The new value.
            test_mean (float): The mean used for distance calculation.

        Returns:
            float: The updated variance.
        """
        try:
            new_n = self.n + 1
            distance_squared = np.square(np.linalg.norm(x - test_mean))
            return ((new_n-1)*self.var + distance_squared)/new_n    
        except Exception as e:
            print(f"Error in get_updated_variance_k_3: {e}")
        
    def get_number_of_points(self):
        """
        Returns the number of points in the datacloud.

        Returns:
            int: The number of points.
        """
        return self.n

    def update_stats(self, new_mean, new_var):
        """
        Updates the mean and variance of the datacloud.

        Parameters:
            new_mean (float): The new mean value.
            new_var (float): The new variance value.
        """
        try:
            self.mean = new_mean
            self.var = new_var
        except Exception as e:
            print(f"Error in update_stats: {e}")
        
    def add_point(self, new_point):
        """
        Adds a point to the datacloud.

        Parameters:
            new_point (hashable): The new point to add.
        """
        try:
            # add the point to the set
            self.points.add(new_point) 
            # update the number of points in the datacloud
            self.n += 1
        except Exception as e:
            print(f"Error in add_point: {e}")
            
    def add_points(self, new_points):
        """
        Adds multiple points to the datacloud.

        Parameters:
            new_points (iterable): The new points to add.
        """
        try:
            # get the union from the point in the cloud and the newly added points
            self.points = self.points.union(new_points)
            # update the number of points in the datacloud
            self.n = len(self.points)
        except Exception as e:
            print(f"Error in add_points: {e}")

    def _adjust_variance(self, new_point):
        """
        Adjust the variance of the data cloud based on a new point.

        Parameters:
        - new_point: The new data point to consider for variance adjustment.
        """
        try:
            # Calculate the Euclidean distance between the new point and the current mean
            distance = np.linalg.norm(new_point - self.mean)

            # Adjust the variance by adding the square of the distance to the current variance
            self.var += distance ** 2

        except Exception as e:
            # Handle exceptions (Specify the exception type you want to catch)
            print(f"Error in adjust_variance: {e}")

    def _calculate_force(self, point):
        """
        Calculate the 'force' based on the distance of a point to the mean.

        Parameters:
        - point: The data point for which to calculate the force.

        Returns:
        - force: The calculated force based on the distance.
        """
        try:
            # Calculate the Euclidean distance between the point and the mean
            distance = np.linalg.norm(point - self.mean)
            force = 1 / (1 + distance)
            return force

        except Exception as e:
            print(f"Error in calculate_force: {e}")
            return 0 

    def _adjust_variance_with_force(self, points):
        """
        Adjust the variance of the data cloud based on the total force calculated from a list of points.

        Parameters:
        - points: List of data points.

        """
        try:
            # Calculate the total force by summing the individual forces
            total_force = sum(self._calculate_force(point) for point in points)

            # Adjust the variance based on the total force
            # One can use total_force directly or apply some function to it
            self.var += total_force

        except Exception as e:
            print(f"Error in adjust_variance_with_force: {e}")

        
class TEDACloud:
    # ------------------------------
    # CONSTRUCTOR
    # ------------------------------

    alfa = np.array([0.0], dtype=float)
    relevanceList = np.zeros([1], dtype=int)

    def __init__(self, m=1, max_clouds=3):
        """
        Initializes a new TEDACloud object.

        Parameters:
            m (int): A parameter (default is 1).
            max_clouds (int): Maximum number of clouds (default is 3).
        """
        try:
            # initialize an empty datacloud array
            self.clouds = np.array([], dtype=DataCloud)
            TEDACloud.classIndex = [[1.0], [1.0]]
            TEDACloud.argMax = []
        except Exception as e:
            print(f"Error initializing TEDACloud: {e}")
    

    # ------------------------------
    # CLOUD METHODS
    # ------------------------------
    def create_cloud(self, x, verbose=False):
        """
        Creates a new datacloud with the given point.

        Parameters:
            x (float): The initial value for mean of the new cloud.
        """
        try:
            # create cloud 1 and add X
            c = DataCloud(x)
            # append the new cloud to the cloud list
            self.clouds = np.append(self.clouds, c)
            self.n_clouds += 1
            if verbose:
                print("New cloud created!")
        except Exception as e:
            print(f"Error in create_cloud: {e}")
        
    def delete_cloud(self, c):
        """
        Deletes a cloud from the TEDACloud.

        Parameters:
            c (DataCloud): The cloud to be deleted.
        """
        try:    
            c_index = np.where(self.clouds == c)
            self.clouds = np.delete(self.clouds, c_index)
        except Exception as e:
            print(f"Error in delete_cloud: {e}")

    def merge_clouds(self, c_target,  c_source):
        """
        Merges two clouds in the TEDACloud.

        Parameters:
            c_target (DataCloud): The target cloud for the merge.
            c_source (DataCloud): The source cloud for the merge.
        """
        try:
            # if one of the clouds is no longer in the list, it has been merged
            if((c_target not in self.clouds) or (c_source not in self.clouds)):
                return
            # else, merge the clouds updating the stats of the resulting cloud and adding the new points
            new_mean = ((c_target.n*c_target.mean)+ (c_source.n*c_source.mean))/(c_target.n + c_source.n)
            new_var = ((c_target.n - 1)*c_target.var + (c_source.n - 1)*c_source.var)/(c_target.n + c_source.n - 2)
            c_target.update_stats(new_mean, new_var)
            c_target.add_points(c_source.points)
            # delete the other cloud from the list
            self.delete_cloud(c_source)
        except Exception as e:
            print(f"Error in merge_clouds: {e}")
        
    def check_clouds_intersection(self):
        """
        Checks for intersection between clouds and merges them if necessary.
        """
        try:
            # copy the list before iterating
            temp_cloud_list = self.clouds.copy()
            
            for _, c1 in enumerate(temp_cloud_list):
                for _, c2 in enumerate(temp_cloud_list):
                    if c1 == c2:
                        continue
                    c1_points = c1.points
                    c2_points = c2.points
                    # check the length of intersection
                    intersect_len = len(c1_points.intersection(c2_points))
                    # get the number of points that are exclusevely in one of the clouds
                    c1_len = len(c1_points) - intersect_len
                    c2_len = len(c2_points) - intersect_len
                    if(intersect_len > c1_len or intersect_len > c2_len):
                        self.merge_clouds(c1, c2)
        except Exception as e:
            print(f"Error in check_clouds_intersection: {e}")
        
    # ------------------------------
    # INTERNAL METHODS
    #-------------------------------
    def save_result(self, df):
        """
        This method is responsible for updating a DataFrame (df) with the assigned data 
        cloud indices for each data point. It iterates over the clouds in the TEDACloud 
        instance and associates each point in the DataFrame with its corresponding data 
        cloud index.

        Parameters:
            df (DataFrame): The DataFrame to be updated.
        
        """
        try:
            cloud_points = [cloud.points for cloud in self.clouds]
            for idx, points in enumerate(cloud_points):
                cloud_number = idx + 1
                for p in points:
                    df_index = int(df[df['time'] == p].index[0])
                    if not pd.isnull(df.loc[df_index, 'datacloud']):
                        original_cloud_list = df.loc[df_index, 'datacloud']
                        df.at[df_index, 'datacloud'] = tuple([*original_cloud_list, cloud_number])
                    df.at[df_index, 'datacloud'] = tuple([cloud_number])
        except Exception as e: 
            print(f"Error in save_result: {e}")

    # ------------------------------
    # EXTERNAL METHODS
    #-------------------------------
    def display_metrics(self):
        """
        Print the metrics of the TEDACloud instance.
        """
        print('Number of DataClouds: {}'.format(len(self.clouds)))

        for i, c in enumerate(self.clouds):
            print('DataCloud {}:'.format(i+1))
            print('Number of points: {}'.format(c.n))
            print('Points: {}'.format(c.points))
            print('Mean: {}'.format(c.mean))
            print('Variance: {}'.format(c.var))
            print('----------------------------------')            

    def print_datacloud_numbers(self):
        """
        Prints the number of DataClouds in the TEDACloud.
        """
        print('Number of DataClouds: {}'.format(self.n_clouds)) 
        
    def run_offline(self, df, features, m, num_cloud=3):
        """
        Runs the TEDA algorithm on the given dataframe

        Parameters:
            df (pd.DataFrame): The DataFrame containing the dataset.
            features (list): The list of feature column names.
            num_cloud (int): Maximum number of clouds (default is 3).
        
        """
        
        # calculate m from the length of the feature list
        self.m = m
        # add datacloud column to the dataframe
        df['time'] = np.nan
        df['datacloud'] = np.nan
        df['datacloud'] = df['datacloud'].astype('object')
        # initialize the time
        k = 1
        #initialize count cloud
        clouds = 0

        # loop through the rows in df
        for index, row in df.iterrows():
            # build the X sample numpy array
            x = np.array(row[features])
            # add k to the dataframe
            df.at[index, 'time'] = k
            if (k == 1):
                # create cloud 1 and add X
                self.create_cloud(x, k)
                TEDACloud.argMax.append(0)
            elif (k == 2):
                c = self.clouds[0]
                TEDACloud.argMax.append(0)
                # calculate the updated cloud stats
                mean = c.get_updated_mean(x)
                var = c.get_updated_variance_k_2(x)
                # update the cloud stats
                c.update_stats(mean, var)
                # add the second point to the first datacloud
                c.add_point(k)
            elif (k >= 3):
                should_create_new_cloud = True
                max_typicality = float("-inf")
                chosen_cloud = -1
                nothing_cloud = True
                TEDACloud.alfa = np.zeros((np.size(self.clouds)),dtype=float)
                # iterate trhough the dataclouds
                for i, c in enumerate (self.clouds):
                    # calculate the new mean and variance of the cloud if we added the point
                    test_n = c.n + 1
                    test_mean = c.get_updated_mean(x)
                    test_var = c.get_updated_variance_k_3(x, test_mean)
                    # calculate the eccentricity and normed eccentricity
                    eccentricity = (test_var + (test_mean - x).T.dot(test_mean - x)) / (test_n * test_var) \
                        if test_var != 0 else float("+inf")
                    norm_eccentricity = eccentricity / 2
                    # calculate the tipicality and normed tipicality
                    typicality = 1 - eccentricity
                    norm_typicality = typicality / (k-2)
                    # check the point pertinence to the cloud
                    is_point_in_c = norm_eccentricity <= (self.m ** 2 + 1) / (2 * test_n)
                    if(is_point_in_c):
                        # update the cloud stats
                        c.update_stats(test_mean, test_var)
                        # add the point to the cloud
                        c.add_point(k)
                        should_create_new_cloud = False
                        nothing_cloud = False
                    elif(norm_typicality >= max_typicality and nothing_cloud and len(self.clouds) == num_cloud):
                        max_typicality = norm_typicality
                        chosen_cloud = i
                        should_create_new_cloud = False
                    TEDACloud.alfa[i] = norm_typicality

                # Check if the point didn't belong to any cloud
                if(should_create_new_cloud and len(self.clouds) < num_cloud):
                    # create a new cloud for the point
                    self.create_cloud(x, k)  
                elif(nothing_cloud and chosen_cloud != -1 and len(self.clouds) == num_cloud):
                    test_mean = self.clouds[chosen_cloud].get_updated_mean(x)
                    test_var = self.clouds[chosen_cloud].get_updated_variance_k_3(x, test_mean)
                    # update the cloud stats
                    self.clouds[chosen_cloud].update_stats(test_mean, test_var)
                    # add the point to the cloud
                    self.clouds[chosen_cloud].add_point(k)
                              
                # check for clouds that can be merged
                self.check_clouds_intersection()

                TEDACloud.relevanceList = TEDACloud.alfa / np.sum(TEDACloud.alfa)
                TEDACloud.argMax.append(np.argmax(TEDACloud.relevanceList))
                TEDACloud.classIndex.append(TEDACloud.alfa)
            
            # update the time
            k += 1
        # once the loop is finished, save the result to the dataframe
        #print([len(cloud.points) for cloud in self.clouds])
        self.save_result(df)


    def run_online(self, point, k, m, num_cloud=3, is_outlier=False):
        """
        Runs the TEDA algorithm on the given dataframe

        Parameters
        ----------
        features : list
            List of features to use for the algorithm
        
        
        """
        
        x = np.array(point)
        
        if (k == 1):
            # Create cloud 1 and add X
            self.create_cloud(x, k)
            return 0
        elif (k == 2):
            self.create_cloud(x, k)
            return 0
        elif (k >= 3):

            if len(self.clouds) < num_cloud:
                self.create_cloud(x, k)
                return len(self.clouds) - 1

            
            max_typicality = float("-inf")
            chosen_cloud = -1
            nothing_cloud = True
            TEDACloud.alfa = np.zeros((np.size(self.clouds)), dtype=float)

            # Iterate trhough the dataclouds
            for i, c in enumerate (self.clouds):
                # Calculate the new mean and variance of the cloud if we added the point
                test_n = c.n + 1
                test_mean = c.get_updated_mean(x)
                test_var = c.get_updated_variance_k_3(x, test_mean)
                # Calculate the eccentricity and normed eccentricity
                eccentricity = (test_var + (test_mean - x).T.dot(test_mean - x)) / (test_n * test_var) if test_var != 0 else float("+inf")
                norm_eccentricity = eccentricity / 2
                # Calculate the tipicality and normed tipicality
                typicality = 1 - eccentricity
                norm_typicality = typicality / (k-2)
                # Check the point pertinence to the cloud
                is_point_in_c = norm_eccentricity <= (m ** 2 + 1) / (2 * test_n)
                if(is_point_in_c):
                    # Update the cloud stats
                    c.update_stats(test_mean, test_var)
                    # Add the point to the cloud
                    c.add_point(k)

                    nothing_cloud = False
                    chosen_cloud = i
                elif norm_typicality >= max_typicality and nothing_cloud:
                    max_typicality = norm_typicality
                    chosen_cloud = i

                TEDACloud.alfa[i] = norm_typicality
            
            if nothing_cloud and is_outlier:
                # get the cloud with the highest typicality
                chosen_cloud = np.argmax(TEDACloud.alfa)
                
                self.clouds[chosen_cloud].mean = x

                # adjust the variance of the cloud
                self.clouds[chosen_cloud]._adjust_variance_with_force(x)

                # add the point to the cloud
                self.clouds[chosen_cloud].add_point(k)
                
            elif chosen_cloud != -1 and is_outlier:
                self.clouds[chosen_cloud].mean = x

                # adjust the variance of the cloud
                self.clouds[chosen_cloud]._adjust_variance_with_force(x)

                # add the point to the cloud
                self.clouds[chosen_cloud].add_point(k)
            elif chosen_cloud != -1:
                # get the cloud with the highest typicality
                chosen_cloud = np.argmax(TEDACloud.alfa)

                test_mean = self.clouds[chosen_cloud].get_updated_mean(x)

                # adjust the variance of the cloud
                self.clouds[chosen_cloud]._adjust_variance_with_force(x)

                # add the point to the cloud
                self.clouds[chosen_cloud].add_point(k)

            return chosen_cloud