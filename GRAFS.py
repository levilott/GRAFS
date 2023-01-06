# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:44:01 2022

@author: Bradford Lott

RFE-GRAFS Feature Selection Implementation Master File

This will be the front facing version on GitHub
"""

###################################################################

"""INSERT PRE-PROCESSING FILE FROM THESIS
   WILL NEED TO ADD RFE CAPABILITY IN THIS SECTION"""


# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:43:23 2021

@author: Bradford Lott

Pre-Processing for my thesis

"""

###################################################################

"""
Created on Thu Sep 23 08:32:25 2021

@author: Bradford Lott

Lott Feature Selection Approach
"""

####################################################################

"TO DO:"

"UPDATE NEW OHE NAMES WITH ORGIINAL VARIBLE NAME AS WELL IN ONE HOT ENCODER FUNCTION"

"We have features and responses stored twice as dataframes and arrays, this is inefficient and could casue problems if we dont update both, just get it working for now and fix in future"

"Currently we wont check for unique row ids, for example if someone includes id as a feature it wont automatically be dropped. Perhaps we should go back and check if a column only contains unique intigers or names."

####################################################################

"Read in file."

def read_data(filepath):
    """
    Parameters
    ----------
    filepath : Path to user data Must be csv, txt, or xls file.

    Returns
    -------
    Dataframe with all variables (features and any response).

    """
    import pandas as pd
    
    if 'csv' in filepath:
        Full_Data = pd.read_csv(filepath)
    
    elif 'txt' in filepath:
            Full_Data = pd.read_csv(filepath)
    
    elif 'xls' in filepath:
        Full_Data = pd.read_excel(filepath)
    
    else: 
        print("Please ensure you have provided a csv, txt, or xls file.")
    
    return Full_Data

#####################################################################
"""Check for response column. Seperate Features and Response."""

def check_response(Data,contains_response,**kwargs):
    """
    
    Parameters
    ----------
    Data : Full dataset as pandas dataframe with all variables (features and response)
    contains_response : Bool, True if contains response else False
        DESCRIPTION.
    response_name : String, response column name

    Returns
    -------
    Seperate dataframes for our feature data and response data

    """
    import pandas as pd
    
    if contains_response == True:
        
        response_name = kwargs.get('response_name', None)
        Response_Data = pd.DataFrame(Data[response_name])
        #make response column Y
        #going to do this for both user input and fabricated responses
        #the user will know what their response column will be called if it contains one
        #need to update some doc strings to explain this though
        Response_Data.columns = ['Y']
        Feature_Data = pd.DataFrame(Data[Data.columns[Data.columns!=response_name]])
        
        return Feature_Data , Response_Data , contains_response
    
    else:
        Feature_Data = Data
        Response_Data = pd.DataFrame()
        print("No response present. Fabricate categorical response using K-Means clustering.")
        
        return Feature_Data , Response_Data, contains_response




######################################################################
        
def RFE(Feature_OHE_Data,Response_Data, perform_rfe,**kwargs):
    """
    
    Need to update this doc string. This performs RFE and uses output as seed
    to GRAFS if True.

    """
    
    
    import pandas as pd
    import numpy as np
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeClassifier
    
    if perform_rfe == True:
        rfe_retain = kwargs.get('rfe_retain', None)
        rfe_retain = int(rfe_retain)
        rfe_step = kwargs.get('rfe_step', 1)
        rfe_step = int(rfe_step)
        
        #categorical multi-category response
        if Response_Data['Y'].dtype == 'int32' or Response_Data['Y'].dtype == 'int64' or Response_Data['Y'].dtype == 'object':
            estimator = DecisionTreeClassifier()
        else:
            estimator = DecisionTreeRegressor()
            

        selector = RFE(estimator, n_features_to_select=rfe_retain, step=rfe_step)
        selector = selector.fit(Feature_OHE_Data, Response_Data)
        selector.support_
        selector.score(Feature_OHE_Data, Response_Data)
        RFE_Features = list(Feature_OHE_Data.columns[selector.support_])
           
        return RFE_Features
    
    else:
        RFE_Features = pd.DataFrame()
        print("RFE not performed.")
        
        return RFE_Features




######################################################################


"NEED TO UPDATE THIS TO INCLUDE THE VARIABLE NAME AS PART OF THE NEW OHE NAME"

"Create One Hots for categorical Features. Can use this same function on Response."

def one_hot_encoder(data):
    """
    Checks whether a dataframe contains categorical data, then performs one-hot encoding on categorical features

    Args:
        data (pandas dataframe): Input dataframe
            
    Returns:
        data (pandas dataframe): Transformed dataframe with one-hot encoded categorical features

    """
    import pandas as pd
    
    #get list of cols which are object data types (text data types)
    categorical_cols = list(data.select_dtypes(include=['object']).columns)

    #apply get dummies from pandas to do one hots over list of categorical features
    one_hots = pd.DataFrame()
    for i in categorical_cols:
        hold = pd.get_dummies(data[i], prefix=f"{i}_One_Hot") #UPDATE WITH NAME HERE
        one_hots = pd.concat([one_hots, hold], axis=1, ignore_index=False)
    
    #cbind one hots to full dataset
    data=pd.concat([data, one_hots], axis=1, ignore_index=False)

    #drop cols with object data type
    OHE_Data = (data.drop(columns=categorical_cols)).astype(float, errors = 'raise')
    
    return OHE_Data

######################################################################

"Check contains response. If False then perform K-means to fabricate categorical response."

def fabricate_response(contains_response, RandomSeed, Standard_Feature_Array, lower_k, upper_k, n_init, max_iter, method):
    """
    Parameters
    ----------
    contains_response : Bool, True if the dataset contains a response taken from check response function
    
    RandomSeed : Random seed to use in K-means
        
    Standard_Feature_Array : Standardized Features
    
    lower_k : Smallest number of groups to consider, should default this to 2
    
    upper_k : Largest number of groups to consider, could default this to some function of the feature space size
    
    n_init : number of initializations to try
    
    max_iter : max iterations
    
    method : string, random initialization or K++, "random", "k-means++"

    Returns
    
    fabricated response data

    """
    
    if contains_response != True:
        
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        import pandas as pd
        
        kmeans_kwargs = {
        "init": method,
        "n_init": n_init,
        "max_iter": max_iter,
        "random_state": RandomSeed,
        }
        
        
        silhouette_coefficients = []
   
 
        for k in range(lower_k, upper_k):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(Standard_Feature_Array)
            score = silhouette_score(Standard_Feature_Array, kmeans.labels_)
            silhouette_coefficients.append(score)
            
        silhouette_coefficients = pd.DataFrame(silhouette_coefficients)
        
        best_k = silhouette_coefficients[0].argmax() + lower_k
        
        kmeans = KMeans(n_clusters = best_k, **kmeans_kwargs)
        kmeans.fit(Standard_Feature_Array)
        
        Fabricated_Response_Array = kmeans.labels_
        
        Fabricated_Response_Array = Fabricated_Response_Array + 1 #add 1 so we dont have a "0" cluster just more intuitive to me
        
        Fabricated_Response_Data = pd.DataFrame(Fabricated_Response_Array)
        
        Fabricated_Response_Data.columns = ['Y']
        
        return silhouette_coefficients, best_k, Fabricated_Response_Data

######################################################################

"If categorical response, create stratified train/validation split based on response"
"We have no need for a test set as we do not care about our models predictive performance"

#not clean to pass features as array and response as data
#i dont like that im doing it this way and should go back and change this
def splitData(Standard_Feature_Array, Response_Data):
    """
    This function creates a test/validation split. It stratifies by response type if categorical.

    Parameters
    ----------
    Standard_Feature_Data : standardized feature data frame
    Response_Data : response data frame

    Returns
    -------
    Balanced test/validation split based on response type if categorical.

    """
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    if Response_Data['Y'].dtype == 'int32' or Response_Data['Y'].dtype == 'int64' or Response_Data['Y'].dtype == 'object':
        x_train, x_val, y_train, y_val = train_test_split(Standard_Feature_Array, Response_Data, 
                                                          test_size=0.4,
                                                          stratify=np.array(Response_Data),
                                                          random_state=1234)
    else:
        x_train, x_val, y_train, y_val = train_test_split(Standard_Feature_Array, Response_Data,
                                                          test_size=0.4,
                                                          shuffle = False,
                                                          stratify = None,
                                                          random_state=1234)
          
    Train_Feature_Data = pd.DataFrame(x_train)
    Val_Feature_Data = pd.DataFrame(x_val)
    
    Train_Response_Data = pd.DataFrame(y_train)
    Val_Response_Data = pd.DataFrame(y_val)
    
    return Train_Feature_Data, Val_Feature_Data, Train_Response_Data, Val_Response_Data
    

######################################################################

"""
Created on Thu Sep 23 11:12:47 2021

@author: Bradford Lott

"""

def FSPreProcessing(filepath, contains_response, perform_rfe, **kwargs):
    """
    Parameters
    ----------
    REQUIRED:
       
    filepath : String, path to datafile
    
    contains_response : Bool, True if dataset contains a response otherwise False
    
    perform_rfe : Bool, if true we will use RFE output to seed GRAFS approach
    
    OPTIONAL:
        
    response_name: String, name of response column
    
    rfe_retain : int, user presribed number of features for RFE to retain

    rfe_step : int, number of features to drop per step in RFE

    RETURNS:
    
    Full_Data:
        
    Train_Feature_Data:
    
    Val_Feature_Data:
        
    Train_Response_Data:
        
    Val_Response_Data:
        
    Train_Response_Vector:
        
    Val_Response_Vector:
    
    
    -------
    
    This funciton will perform all pre-processing steps to format data for our FS algorithm.
    These steps include:
        
        Read in data from the provided file path
        
        If the dataset does not contain a response perform K-means++ to fabricate a response...
        ...choosing K-clusters where K is the best K based on silhouette score
        
        Create one-hot encodings for categorical features and responses
        
        Standardize our feature space
        
        Create train and validation splits, stratified by response if response is categorical
        
        Format our response data as vector for our NN

    """
    
    response_name = kwargs.get('response_name', None)
    rfe_retain = kwargs.get('rfe_retain', 12)
    rfe_retain = int(rfe_retain)
    rfe_step = kwargs.get('rfe_step', 1)
    rfe_step = int(rfe_step)
    
    #import our functions file
    import pandas as pd
    import numpy as np
    
    #read data funciton from FS Functions
    Full_Data = read_data(filepath)
    
    #check response and seperate response and features
    Feature_Data, Response_Data, contains_response = check_response(Full_Data, contains_response = contains_response, response_name = response_name)
    
    #one hot encode categorical features
    Feature_OHE_Data = one_hot_encoder(Feature_Data)
    #this will return one hot encoded data as OHE_Data
    
    #perform RFE here if desired and return as Feature_OHE_Data to maintain
    #naming convetion for easy integration into rest of the code
    
    if perform_rfe == True:
        RFE_Features = RFE(Feature_OHE_Data, Response_Data, perform_rfe = perform_rfe, rfe_retain = rfe_retain, rfe_step = rfe_step)
        Feature_OHE_Data = Feature_OHE_Data[RFE_Features]
    
    #Standardize Feature Data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Standard_MainEffects_Array = scaler.fit_transform(Feature_OHE_Data)
    
    
    #if contains_response from check response == True then we already have our resposne column
    #otherwise we need to fabricate a categorical response from K-means clustering
    if contains_response != True:
       silhouette_coefficients, best_k, Response_Data = fabricate_response(contains_response = contains_response, RandomSeed = 1234, Standard_Feature_Array = Standard_MainEffects_Array, lower_k = 2, upper_k = 11, n_init = 10, max_iter = 300, method = "k-means++")

    
    ################################################################3
    """INSERT INTERACTIONS AND CORRELATIONS CALCULATIONS"""
    
    from itertools import combinations
    #establish all as empty dataframes
    Interactions_Data = pd.DataFrame()
    
    Interaction_Terms = pd.DataFrame()
    Main_Terms = pd.DataFrame()
    Square_Terms = pd.DataFrame()
    
    #capture interaction data for all interactions
    for col1, col2 in combinations(Feature_OHE_Data.columns, 2):
        Interactions_Data[f"{col1}_{col2}"] = Feature_OHE_Data[col1] * Feature_OHE_Data[col2]
    
    #capture interaction term names, type, main effects
    Interaction_Terms['Feature'] = Interactions_Data.columns
    Interaction_Terms['Type'] = "Interaction"
    
    Effect_List = []
    for col1, col2 in combinations(Feature_OHE_Data.columns, 2):
        Effect_List.append([f"{col1}", f"{col2}"])
    
    Interaction_Terms['MainEffects'] = Effect_List  
    
    #capture main effect and squared data
    for col in Feature_OHE_Data.columns:
        Interactions_Data[f"{col}"] = Feature_OHE_Data[col]
        Interactions_Data[f"{col}_{col}"] = Feature_OHE_Data[col]**2
    
    #capture response
    Interactions_Data['Y'] = Response_Data['Y']
    
    
    #capture main effect term names, type, main effects 
    Main_Terms['Feature'] = Feature_OHE_Data.columns
    Main_Terms['Type'] = "Main"
    
    Effect_List = []
    for name in Feature_OHE_Data.columns:
        Effect_List.append([name])
        
    Main_Terms['MainEffects'] =  Effect_List
    
    #capture square term names, type, main effects   
    Effect_List = []
    for name in Feature_OHE_Data.columns:
        Effect_List.append(f"{name}_{name}")
    Square_Terms['Feature'] = Effect_List
    Square_Terms['Type'] = "Square"
    
    #not efficient to do this over but it works for now
    #could assign based on previous dataframe but dont want to for now
    
    Effect_List = []
    for name in Feature_OHE_Data.columns:
        Effect_List.append([name])
    
    Square_Terms['MainEffects'] = Effect_List
    
    All_Terms = pd.concat([Main_Terms, Interaction_Terms, Square_Terms], axis = 0, ignore_index = True)
    All_Terms.reset_index(drop = True, inplace = True)
    
    #Interactions_Data.to_csv('Interactions_Test.csv', index = False)
    
    #make our correlations for numeric response
    if Response_Data['Y'].dtype == 'float64':
        Correlations = pd.DataFrame(Interactions_Data[Interactions_Data.columns[0:]].corr()['Y'])
        Correlations.reset_index(level = 0, inplace = True)
        Correlations.columns = ['Feature', 'Corr']
        drop_rows = Correlations[ Correlations['Feature'] == 'Y' ].index
        Correlations.drop(drop_rows, inplace = True)
        Correlations['Corr'] = Correlations['Corr'].abs()
    else:
        Interactions_Corr = one_hot_encoder(Interactions_Data)
        response_level_count = len(Response_Data['Y'].unique())
        HoldCorrs = pd.DataFrame(Interactions_Corr[Interactions_Corr.columns[0:]].corr().iloc[:,-response_level_count:])
        HoldCorrs.drop(HoldCorrs.tail(response_level_count).index,inplace = True)
        HoldCorrs = HoldCorrs.abs()
        Correlations = pd.DataFrame()
        Correlations['Corr'] = HoldCorrs.max(axis=1)
        Correlations.reset_index(level = 0, inplace = True)
        Correlations.columns = ['Feature', 'Corr']
        
    
    """Correlations for one hots from the same variable appear to be NaN which works out well.
    We can drop these"""
    
    Correlations = Correlations.dropna(axis= 0)
    
    """Need to sort next"""
    
    """Add correlation of all featuers so we can pull later in feature addition
    see notes next to add feature in LottFSNN file"""
    
    Correlations = Correlations.sort_values(by = ['Corr'], ascending = False, inplace = False)
    
    Correlations.reset_index(drop = True, inplace = True)
    
    #merge Correlations with All_Terms to get list of effects which make up each term
    #this list will be passed to actively select the terms we want in each iteration
    
    Correlations = Correlations.merge(All_Terms, how = 'left', on = "Feature")
    
    MECorr = Feature_OHE_Data.corr()
    
    #drop Y from interactions data to use in creating feature training data
    #cant drop before this since we use Y in interactions data for correlations
    Interactions_Data.drop('Y', axis = 1, inplace = True)    
    
    ###########################################################
    
    #Standardize Feature Data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Standard_Feature_Array = scaler.fit_transform(Interactions_Data)
    
    
        
    #may also want to normalize response data and compare
    #standardize resposne data
    if Response_Data['Y'].dtype == 'float64':
        Response_Data = pd.DataFrame(scaler.fit_transform(Response_Data))
        Response_Data.columns = ['Y']
    
    #now that we have data we can create a train/validation split
    #we have no need for test data here since we do not care about our models predictive performance
    
    Train_Feature_Data, Val_Feature_Data, Train_Response_Data, Val_Response_Data = splitData(Standard_Feature_Array = Standard_Feature_Array, Response_Data = Response_Data)
    
    #update column names
    Train_Feature_Data.columns = Interactions_Data.columns
    Val_Feature_Data.columns = Interactions_Data.columns
    
    #Response_Data[0].dtype       
       
    #check response type
    #if its continuous we'll standardize it
    #if its categorical we'll create one hot vectors
    
    #THIS ASSUMES ANY INTEGER RESPONSE WILL BE A CATEGORICAL VARIABLE
    #NUMBERS WITHOUT DECIMALS WOULD BE TREATED AS CATEGORICAL WHICH IS NOT GREAT
    #TO BE FAIR MOST PACKAGES ARE NOT INTUITIVE ABOUT THIS EITHER
    #SHOULD GO BACK AND UPDATE FOR USABILITY   
    
    if Train_Response_Data['Y'].dtype == 'int32' or Train_Response_Data['Y'].dtype == 'int64' or Train_Response_Data['Y'].dtype == 'object':  
        Train_Response_Vector = pd.get_dummies(Train_Response_Data['Y'].values)   
        Train_Response_Vector = np.asmatrix(Train_Response_Vector)
        Val_Response_Vector = pd.get_dummies(Val_Response_Data['Y'].values) 
        Val_Response_Vector = np.asmatrix(Val_Response_Vector)
    else:
        Train_Response_Vector = (np.asmatrix(Train_Response_Data['Y'].values)).transpose()
        Val_Response_Vector = (np.asmatrix(Val_Response_Data['Y'].values)).transpose()
 
####################################################################################
            
        
    """Search Strategy"""
    
    """Fabricate all interactions and squared terms, pre-process including one hots and standardization
    then run correlation matrix including response. Check all features, interactions, and squared terms
    against the response. If response is categorical we'll have to consider all levels of the response."""
    
    
    #import pandas as pd
    
    
    """Its a waste of memory / computational time to calculate squared terms for or interactions
    between different levels for the same one hot encoded variable. We still want to capture interactions between one hots and
    continuous variables. Not sure how we want to do this with the current code though. Its still
    running pretty quick and wont hurt any of the results but we could probably speed it up a bit
    here if we look more into this. But also to play devils advoate, taking the time to search through
    and identify which variables we should calculate/keep may take longer than just multiplying some
    ones and zeros..."""
    
    
    """NEED TO FIGURE OUT HOW TO CAPTURE FEATURE NAME, TYPE, MAINEFFECTS, AND CORR
    IN ONE DATAFRAME. MIGHT USE A DICT. USING _TERMS FOR TERM NAMES AND INTERACTIONS_DATA
    FOR THE CORRELATIONS. THE CORRELATIONS ARE WORKING. THE TERMS ARE NOT"""
    
    #make Interactions_Data which will contain the numeric values for our interactions
    #these numeric values will be used to calculate correlations
    
    #make _Terms which will capture the names and associated features for all... 
    #...interactions/maineffects/square terms
    
    
    
    return Full_Data, MECorr, Correlations, Train_Feature_Data, Val_Feature_Data, Train_Response_Data, Val_Response_Data, Train_Response_Vector, Val_Response_Vector


#############################################################################
    
"""INSERT FSNN CODE"""

##############################################################################
def LottFSNN(Full_Data, MECorr, Correlations, Train_Feature_Data, Val_Feature_Data, Train_Response_Data, Val_Response_Data, Train_Response_Vector, Val_Response_Vector, **kwargs):
    import numpy as np
    import pandas as pd
    #import tensorflow as tf
    #from tensorflow import keras
    #from tensorflow.keras import layers
    
    from keras.models import Model
    #from keras.models import Sequential #uncomment if we use sequential method
    from keras.optimizers import SGD
    from keras.layers import Dense
    from keras.layers import Input
    from keras.utils.np_utils import to_categorical
    
    if Train_Response_Data['Y'].dtype == 'int32' or Train_Response_Data['Y'].dtype == 'int64' or Train_Response_Data['Y'].dtype == 'object':
        eta = kwargs.get('eta', 10)
        eta = int(eta)
        epsilon = kwargs.get('epsilon', 10)
        epsilon = int(epsilon)
        delta = kwargs.get('delta', 3)
        delta = int(delta)
        xi = kwargs.get('xi', 0.8)
        xi = float(xi)
        alpha = kwargs.get('alpha', 0.95)
        alpha = float(alpha)
        phi = kwargs.get('phi', 0.9)
        phi = float(phi)
        omega = kwargs.get('omega', 0.01)
        omega = float(omega)
    #else continuous    
    else:
        eta = kwargs.get('eta', 10)
        eta = int(eta)
        epsilon = kwargs.get('epsilon', 10)
        epsilon = int(epsilon)
        delta = kwargs.get('delta', 3)
        delta = int(delta)
        xi = kwargs.get('xi', 0.8)
        xi = float(xi)
        alpha = kwargs.get('alpha', 0.02)
        alpha = float(alpha)
        phi = kwargs.get('phi', 1.1)
        phi = float(phi)
        omega = kwargs.get('omega', 0.01)
        omega = float(omega)
    
    
    #######################################################
    """Count features/responses"""
    
    
    #can use import math then math.comb in python version 3.8+
    #were in python 3.7 so well just define comb manually
    def comb(N,k):
        import math
        f = math.factorial
        amount = f(N) // (f(k) * f(N-k))
        return amount
    
    
    def countfeatures(Data):
        #capture our number of features/interactions/response
        #maybe go back and add cap at 1225 interactions
        #see notes 
        #it didnt work well
        #see experiment 2 v 2
        num_features = len(Data.columns)
        if num_features > 5:
            num_interactions = comb(num_features, 2)
        else:
            #this says num_interactions but it really represents the minimum number of nodes in our network
            num_interactions = eta
        return num_features, num_interactions
    
    #########################################################
    """Build NN Format"""
    
    def buildNN(num_features, num_interactions, num_response, act_fun, out_act_fun):
        
        inputs = Input(shape=(num_features,))
        x = Dense(num_features, activation= act_fun)(inputs)
        x = Dense(num_interactions, activation= act_fun)(x)
        x = Dense(num_interactions, activation= act_fun)(x)
        outputs = Dense(num_response, activation= out_act_fun)(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    #########################################################
    
    """Iterate features through NN"""
    
    #########################################################
    
    """Only need to do these steps once"""
    #ADJUSTING TO 15 TO TEST AGAINST RFE
    num_epochs = epsilon
    
    #only need to capture num_response once        
    num_response = np.shape(Train_Response_Vector)[1]
    
    """Define acc and loss metrics"""
    #continuous standardized response
    if Train_Response_Data['Y'].dtype == 'float64':
        act_fun = 'relu' #should also try sigmoid
        out_act_fun = 'linear' #will try norm and relu later
        nn_loss = 'mean_squared_error'
        nn_acc = 'mean_squared_error'
    
    #categorical multi-category response
    if Train_Response_Data['Y'].dtype == 'int32' or Train_Response_Data['Y'].dtype == 'int64' or Train_Response_Data['Y'].dtype == 'object':
        if num_response > 1:
            act_fun = 'relu' #should also try sigmoid
            out_act_fun = 'softmax'
            nn_loss = 'categorical_crossentropy'
            nn_acc = 'accuracy'
    
    #see notes in draft for binary options.
    
    #########################################################
    
    #establish lists for tracking, run_count, features, acc
            
    Run_List = []
    Feature_List = []
    Acc_List = []
    
    
    ##########################################################
    
    """Begin baseline"""
    """Baseline will train on all main effects"""
    
    Main_Effects_List = list(MECorr.columns)
    
    num_features, num_interactions = countfeatures(Train_Feature_Data[Main_Effects_List])
    
    model = buildNN(num_features = num_features, num_interactions = num_interactions, num_response = num_response, act_fun = act_fun, out_act_fun = out_act_fun)
    
    model_keep = buildNN(num_features = num_features, num_interactions = num_interactions, num_response = num_response, act_fun = act_fun, out_act_fun = out_act_fun)
    
        
    # complile the model with loss = binary_crossentropy
    model.compile(optimizer= SGD(lr=.01),
                  loss= nn_loss,
                  metrics=[nn_acc])
    
    model_keep.compile(optimizer= SGD(lr=.01),
                  loss= nn_loss,
                  metrics=[nn_acc])
    
    #train NN for some number of epochs
    history_baseline = model.fit(x=Train_Feature_Data[Main_Effects_List],y=Train_Response_Vector, batch_size=1, epochs=num_epochs, verbose= 1, callbacks=None,
                        validation_data=(Val_Feature_Data[Main_Effects_List],Val_Response_Vector), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
    
    Baseline_Data = pd.DataFrame(history_baseline.history)
    
    # history_keep = model_keep.fit(x=Train_Feature_Data[Main_Effects_List],y=Train_Response_Vector, batch_size=1, epochs=num_epochs, verbose= 1, callbacks=None,
    #                     validation_data=(Val_Feature_Data[Main_Effects_List],Val_Response_Vector), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
    
    #Keep_Data = pd.DataFrame(history_keep.history)
    
    # Test_Set = pd.read_csv('Test_Set.csv')
    # Test_Set.drop('Cumulative_Error', axis = 1, inplace = True)
    
    #Keep_Predictions = model_keep.predict(Test_Set)
    
    #define acc_metric to capture from dataframe
    
    #this wont be efficient to do every loop
    #rethink this setup
    if nn_acc == 'mean_squared_error':
        val_acc_metric = 'val_mean_squared_error'
        baseline_acc = Baseline_Data[val_acc_metric].agg(lambda grp: grp.nsmallest(3).mean())
    else:
        val_acc_metric = 'val_accuracy'
        baseline_acc = Baseline_Data[val_acc_metric].agg(lambda grp: grp.nlargest(3).mean())
    
    print('')
    print('With Features:', Main_Effects_List)
    print('Baseline Performance:', val_acc_metric, baseline_acc)
    
    Run_List.append([1])
    Feature_List.append(Main_Effects_List)
    Acc_List.append([baseline_acc])
    
    #assign best_acc to our baseline_acc
    
    best_acc = baseline_acc
    ###############################################################################
    
    """Begin first loop adding from Correlations"""
    
    #will need to loop over .iloc[i] while we add terms
    #this is giving us the main effects for the term with the highest correlation which may be an interaction or squared term
    #see ['Feature'] for the feature with the highest correlations

    Salient_Feature_Train_Data = pd.DataFrame(Train_Feature_Data[Correlations.iloc[0]['Feature']])
    Salient_Feature_Val_Data = pd.DataFrame(Val_Feature_Data[Correlations.iloc[0]['Feature']])
    Current_Feature_List = list(Salient_Feature_Train_Data.columns)
    
    if Correlations.iloc[0]['Type'] == "Interaction":
        ffcorr = MECorr.iloc[MECorr.columns.get_loc(Correlations.iloc[0]['MainEffects'][0])][MECorr.columns.get_loc(Correlations.iloc[0]['MainEffects'][1])]    
        if ffcorr < xi:
            for x in Correlations.iloc[0]['MainEffects']:
                if x not in Current_Feature_List:
                    Current_Feature_List.append(x)
        else:
            Current_Feature_List.append(Correlations.iloc[0]['MainEffects'][0])
    else:
        for x in Correlations.iloc[0]['MainEffects']:
                if x not in Current_Feature_List:
                    Current_Feature_List.append(x)
    
    Salient_Feature_Train_Data = Train_Feature_Data[Current_Feature_List]
    Salient_Feature_Val_Data = Val_Feature_Data[Current_Feature_List]
    
    
    num_features, num_interactions = countfeatures(Salient_Feature_Train_Data)
    
    model = buildNN(num_features = num_features, num_interactions = num_interactions, num_response = num_response, act_fun = act_fun, out_act_fun = out_act_fun)
            
    # complile the model with loss = binary_crossentropy
    model.compile(optimizer= SGD(lr=.01),
                  loss= nn_loss,
                  metrics=[nn_acc])
    
    #train NN for some number of epochs
    history_firstrun = model.fit(x=Salient_Feature_Train_Data,y=Train_Response_Vector, batch_size=1, epochs=num_epochs, verbose= 1, callbacks=None,
                        validation_data=(Salient_Feature_Val_Data,Val_Response_Vector), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
    
    Firstrun_Data = pd.DataFrame(history_firstrun.history)
    
    #define acc_metric to capture from dataframe
    
    #this wont be efficient to do every loop
    #rethink this setup
    if nn_acc == 'mean_squared_error':
        val_acc_metric = 'val_mean_squared_error'
        firstrun_acc = Firstrun_Data[val_acc_metric].agg(lambda grp: grp.nsmallest(delta).mean())
    else:
        val_acc_metric = 'val_accuracy'
        firstrun_acc = Firstrun_Data[val_acc_metric].agg(lambda grp: grp.nlargest(delta).mean())
    
    print('')
    print('With Features:', list(Salient_Feature_Train_Data.columns))
    print('Firstrun Performance:', val_acc_metric, firstrun_acc)
    
    Run_List.append([2])
    Feature_List.append(list(Salient_Feature_Train_Data.columns))
    Acc_List.append([firstrun_acc])
    
    if nn_acc == 'mean_squared_error':
        if firstrun_acc < best_acc:
            best_acc = firstrun_acc
    else:
        if firstrun_acc > best_acc:
            best_acc = firstrun_acc
            
    ########################################################################
            
    """Automate Starting On Second Run"""
    
    #define function to add new features from correlation list
    #will add single or group of features based on interaction
    
    """Add logic to take correlation of new feature with existing features in the model
    check the difference between this value and its correlation with the response"""
    
    #VERSION 1
    def addnewfeature():
        Current_Feature_List = list(Salient_Feature_Train_Data.columns)
        
        num_new_features = 0
        num_current_features = len(Current_Feature_List)
        
        #while num_new_features < 1:
        for i in range(1,len(Correlations)):
            num_new_features = len(Current_Feature_List) - num_current_features
            if num_new_features > 0:
                if Correlations.iloc[i-1]['Feature'] not in Current_Feature_List:
                    Current_Feature_List.append(Correlations.iloc[i-1]['Feature'])
                break
            else:
                for x in Correlations.iloc[i]['MainEffects']:
                    if x not in Current_Feature_List:
                        #add another if statment representing difference between
                        #worst_feature_feature correlation and feature_response correlation
                        #max(MECorr.iloc[MECorr.columns.get_loc('XTWO')][['XTWO','XONE','XTHREE','XSIX']])
                        ffcorr = max(MECorr.iloc[MECorr.columns.get_loc(x)][list(set(Current_Feature_List).intersection(Main_Effects_List))])
                        if ffcorr < xi: #dont add features if they are highly correlated with any feature in our current feature set
                            Current_Feature_List.append(x)
    
        return Current_Feature_List
    
    # #VERSION 2
    # def addnewfeature():
    #     num_new_features = 0
    #     num_current_features = len(Current_Feature_List)
        
    #     #while num_new_features < 1:
    #     for i in range(1,len(Correlations)):
    #         num_new_features = len(Current_Feature_List) - num_current_features
            
    #         if num_new_features > 0:
    #             break
            
    #         #maintaining model hierarchy by adding the main effect for a square
    #         #as well as the square will keep us from missing it later due to
    #         #our correlation check and shouldnt hurt our answer
    #         if Correlations.iloc[i]['Type'] == 'Square':
    #             x = Correlations.iloc[i]['MainEffects']
    #             if f'{x[0]}_{x[0]}' not in Current_Feature_List:
    #                 Current_Feature_List.append(f'{x[0]}_{x[0]}')
    #                 if x not in Current_Feature_List:
    #                     Current_Feature_List.append(x[0])
    #                 #now this is a little different
    #                 #we are adding columns to our training data which
    #                 #were not used in our baseline, this means our baseline may not
    #                 #be representative but it should still improve our performance
    #                 #in the case in which we have a quadratic effect present
    #                 Train_Feature_Data[f'{x[0]}_{x[0]}'] = Train_Feature_Data[x]**2
    #                 Val_Feature_Data[f'{x[0]}_{x[0]}'] = Val_Feature_Data[x]**2
    #                 Salient_Feature_Train_Data = Train_Feature_Data[Current_Feature_List]
    #                 Salient_Feature_Val_Data = Val_Feature_Data[Current_Feature_List]
        
                    
    #         else:
    #             for x in Correlations.iloc[i]['MainEffects']:
    #                 if x not in Current_Feature_List:
    #                     #add another if statment representing difference between
    #                     #worst_feature_feature correlation and feature_response correlation
    #                     #max(MECorr.iloc[MECorr.columns.get_loc('XTWO')][['XTWO','XONE','XTHREE','XSIX']])
    #                     ffcorr = max(MECorr.iloc[MECorr.columns.get_loc(x)][Current_Feature_List])
    #                     if ffcorr < 0.8: #dont add features if they are highly correlated with any feature in our current feature set
    #                         Current_Feature_List.append(x)
    #                         Salient_Feature_Train_Data = Train_Feature_Data[Current_Feature_List]
    #                         Salient_Feature_Val_Data = Val_Feature_Data[Current_Feature_List]
        
    
    #     return Current_Feature_List, Salient_Feature_Train_Data, Salient_Feature_Val_Data
             

    #LOOP
    #THESE STEPS SEEM TO WORK WELL
    #JUST NEED TO SET UP IN LOOP
    
    Stopping_Criteria = False #stopping criteria to exit algorithm
    Degraded_Performance = False #marker if we got worse than our best assuming our best is not our baseline
    #will use degraded performance to back track to correct set if True
    run_count = 3
    
    if nn_acc == 'mean_squared_error':
        if firstrun_acc < alpha:
            Stopping_Criteria = True
    
    if nn_acc == 'accuracy':
        #ADJUSTING TO ADJUST ALPHA
        if firstrun_acc > alpha:
            Stopping_Criteria = True
    
    run_acc = 0.5
    #assign run_acc to 0.5 because it wont cause the loop to quit...
    #...for regression or classification problems on the first loop
    
    """BEGIN LOOP"""
    
    for i in range(1,len(Correlations)):
        
        if nn_acc == 'mean_squared_error':
            if run_acc < alpha:
                Stopping_Criteria = True
        
        if nn_acc == 'accuracy':
            #UPDATING THIS FROM 0.95 TO 0.96 TO TEST BREAST CANCER ASSERTION
            if run_acc > alpha:
                Stopping_Criteria = True
        
        #old method
        #if nn_acc == 'mean_squared_error':
        #    if best_acc != baseline_acc:
        #        if best_acc < 0.01:
        #            Stopping_Criteria = True
        
        #old method
        #if nn_acc == 'accuracy':
        #    if best_acc != baseline_acc: #in the event the baseline and first run both reach 1.0 acc then this doesnt work and will keep running
        #        if best_acc > 0.95:
        #            Stopping_Criteria = True
        
        if Stopping_Criteria == True:
            break 
            
        #USE IF USING addnewfeature VERSION 1
        Current_Feature_List = addnewfeature()
        
        #USE IF USING addnewfeature VERSION 1
        Salient_Feature_Train_Data = Train_Feature_Data[Current_Feature_List]
        Salient_Feature_Val_Data = Val_Feature_Data[Current_Feature_List]
        
        #USE IF USING addnewfeature VERSION 2
        #Current_Feature_List, Salient_Feature_Train_Data, Salient_Feature_Val_Data = addnewfeature()
        
        num_features, num_interactions = countfeatures(Salient_Feature_Train_Data)
        
        model = buildNN(num_features = num_features, num_interactions = num_interactions, num_response = num_response, act_fun = act_fun, out_act_fun = out_act_fun)
                
        # complile the model with loss = binary_crossentropy
        model.compile(optimizer= SGD(lr=.01),
                      loss= nn_loss,
                      metrics=[nn_acc])
        
        history = model.fit(x=Salient_Feature_Train_Data,y=Train_Response_Vector, batch_size=1, epochs=num_epochs, verbose= 1, callbacks=None,
                            validation_data=(Salient_Feature_Val_Data,Val_Response_Vector), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
        
        History_Data = pd.DataFrame(history.history)
        
        if nn_acc == 'mean_squared_error':
            val_acc_metric = 'val_mean_squared_error'
            run_acc = History_Data[val_acc_metric].agg(lambda grp: grp.nsmallest(delta).mean())
        else:
            val_acc_metric = 'val_accuracy'
            run_acc = History_Data[val_acc_metric].agg(lambda grp: grp.nlargest(delta).mean())
        
        #update this to capture in list to be converted into datafarme
        print('')
        print('With Features:', Current_Feature_List)
        print('Run ',run_count,' Performance:', val_acc_metric, run_acc)
        
        Run_List.append([run_count])
        Feature_List.append(Current_Feature_List)
        Acc_List.append([run_acc])
        
        #update best run and stopping criteria
        if nn_acc == 'mean_squared_error': #if numeric response
            if run_acc < best_acc: #if this run is better than our best
                best_acc = run_acc #establish as new best
            elif run_acc > best_acc: #if this run is worse than our best
                if best_acc < baseline_acc: #if we've improved since baseline
                    if run_acc > best_acc: #if this run is worse than our best
                        Stopping_Criteria = True #stop
                        Degraded_Performance = True
                elif best_acc == baseline_acc:#otherwise if our baseline is our best
                    if run_acc < (phi * best_acc):#if we have built close to baseline acc
                        Stopping_Criteria = True#stop
        else: #if categorical response
            if run_acc > best_acc: #if this run is better than our best
                best_acc = run_acc #establish as new best
            elif run_acc < best_acc: #if this run is worse than our best
                if best_acc > baseline_acc: #if we've improved since baseline
                    if run_acc < best_acc: #if this run is worse than our best
                        Stopping_Criteria = True #stop
                        Degraded_Performance = True
                elif best_acc == baseline_acc: #if baseline is our best
                    #UPDATING THIS FROM 0.9 TO 1.0 TO TEST BREAST CANCER ASSERTION
                    if run_acc > (phi * best_acc): #if we have built close to baseline acc
                        Stopping_Criteria = True #stop
        
        run_count +=1
        i +=1
        
    
    All_Runs_Data = pd.DataFrame()
    All_Runs_Data['Run'] = Run_List
    All_Runs_Data['Features'] = Feature_List
    
    if nn_acc == 'mean_squared_error':
        All_Runs_Data['Val_MSE'] = Acc_List
    else:
        All_Runs_Data['Val_Acc'] = Acc_List
        
    
    print('')
    print('All Runs')
    print(All_Runs_Data)
    #ran into a case where we did degraded but by less than omega (0.01 by default) and the code throws an error of course need to program some type of warning here
    if Degraded_Performance == False:
        Final_Features_List = list(set(All_Runs_Data.iloc[-1]['Features']).intersection(Main_Effects_List))
    else:
        for i in range(-2,-len(All_Runs_Data),-1):
            if nn_acc == 'mean_squared_error':
                check_improvement = All_Runs_Data.iloc[i-1]['Val_MSE'][0] - All_Runs_Data.iloc[i]['Val_MSE'][0]
                if check_improvement > omega:
                    last_improvement_index = i
                    break
            if nn_acc == 'accuracy':
                check_improvement = All_Runs_Data.iloc[i]['Val_Acc'][0] - All_Runs_Data.iloc[i-1]['Val_Acc'][0]
                if check_improvement > omega:
                    last_improvement_index = i
                    break
        Final_Features_List = list(set(All_Runs_Data.iloc[last_improvement_index]['Features']).intersection(Main_Effects_List))
        
    return All_Runs_Data, Final_Features_List
# """Need to introduce K-fold by running maybe 10 times and capturing last run
# last run will work since we'll always add features in the same order
# as long as your run was ran maybe 5 or more times then we pick you.
# This is another hyper parameter. Maybe describe as confidence for inclusion 
# out of 10 runs. Let the user choose the number of runs and confidence."""

"""********************END FSNN CODE*********************"""

########################################################################

"""GRAFS PUT IT ALL TOGETHER"""

########################################################################

def GRAFS(filepath, contains_response, perform_rfe, **kwargs):
    """
    BUILD DOC STRING

    Parameters
    ----------
    filepath : TYPE
        DESCRIPTION.
    contains_response : TYPE
        DESCRIPTION.
    perform_rfe : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    import pandas as pd
    import numpy as np
    
    #establish optional arguments through pre-processing
    response_name = kwargs.get('response_name', None)
    rfe_retain = kwargs.get('rfe_retain', 12)
    rfe_retain = int(rfe_retain)
    rfe_step = kwargs.get('rfe_step', 1)
    rfe_step = int(rfe_step)


    #PERFORM PRE-PROCESSING: This includes K-means and RFE as appropriate
    Full_Data, MECorr, Correlations, Train_Feature_Data, Val_Feature_Data, Train_Response_Data, Val_Response_Data, Train_Response_Vector, Val_Response_Vector = FSPreProcessing(filepath = filepath, contains_response = contains_response, response_name = response_name, perform_rfe = perform_rfe, rfe_retain = rfe_retain, rfe_step = rfe_step)
    
    
    #establish option arguments for NN: remember 2 sets based on problem type
    #reference Table 2 Hyper-parameter Settings in publication
    
    #if categorical
    if Train_Response_Data['Y'].dtype == 'int32' or Train_Response_Data['Y'].dtype == 'int64' or Train_Response_Data['Y'].dtype == 'object':
        eta = kwargs.get('eta', 10)
        eta = int(eta)
        epsilon = kwargs.get('epsilon', 10)
        epsilon = int(epsilon)
        delta = kwargs.get('delta', 3)
        delta = int(delta)
        xi = kwargs.get('xi', 0.8)
        xi = float(xi)
        alpha = kwargs.get('alpha', 0.95)
        alpha = float(alpha)
        phi = kwargs.get('phi', 0.9)
        phi = float(phi)
        omega = kwargs.get('omega', 0.01)
        omega = float(omega)
    #else continuous    
    else:
        eta = kwargs.get('eta', 10)
        eta = int(eta)
        epsilon = kwargs.get('epsilon', 10)
        epsilon = int(epsilon)
        delta = kwargs.get('delta', 3)
        delta = int(delta)
        xi = kwargs.get('xi', 0.8)
        xi = float(xi)
        alpha = kwargs.get('alpha', 0.02)
        alpha = float(alpha)
        phi = kwargs.get('phi', 1.1)
        phi = float(phi)
        omega = kwargs.get('omega', 0.01)
        omega = float(omega)
    
    Full_Progress, Salient_Features = LottFSNN(Full_Data, MECorr, Correlations, Train_Feature_Data, Val_Feature_Data, Train_Response_Data, Val_Response_Data, Train_Response_Vector, Val_Response_Vector, eta = eta, epsilon = epsilon, delta = delta, xi = xi, alpha = alpha, phi = phi, omega = omega)
    
    return Full_Progress, Salient_Features

#############################################################################

"""END GRAFS CODE"""

#############################################################################















