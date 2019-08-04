import pandas as pd    
def openDataFile(fileName) :  
    """ Open dataset file and populate X, Y, and E
    Args: 
        fileName (String) : filename of dataset, a structured (CSV) dataset where
                - The first N-2 columns are the features (X).  
                - The next to last column is the label (Y) {0, 1}
                - The last column gives the explanations (E) {0, 1, ..., MaxE}.  We assume the explanation space
                    is dense, i.e., if there are MaxE+1 unique explanations, they will be given IDs from 0 .. MaxE
                - first row contains header information for each column and should be "Y" for labels and "E" for explanations
                - each row is an instance 
    Returns:
            X : list of features vectors
            Y : list of labels
            E : list of explanations
    """                           
    data = pd.read_csv(fileName)     # Load datafile into a dataframe
    X = data.iloc[:,:-2]   # Choose all rows and all cols, except for the last 2 cols
    Y = data['Y']          # Choose col with header 'Y'
    E = data['E']          # Choose col with header 'E'

    return X, Y, E