import re
from tqdm import tqdm
import pandas as pd
import os
# Correcting the dictionaries by adding missing curly braces and creating a list of dictionaries
def read_all_metadata_files():
    '''
    This function reads all the movie metadata files from 1980 - 2016 and generates a list of python dictionaries.
    here each dictionary represent information about one movie poster.
    output: a list of python dictionaries
    '''
    list_of_dicts = [] #to store dictionaries
    
    for file_no in range(1980,2016):
        file_path = 'G:\\OneDrive - Knights - University of Central Florida\\Course Work\\Fall 23\\CAP-5415-CV\\project\\movie dataset\\Movie_Poster_Metadata\\groundtruth\\{}.txt'.format(file_no)
        #print(file_path)
        # the following code is required because text files after 1982 had utf-16 encoding
        if file_no < 1982:
            with open(file_path, 'r') as file:
                data = file.read()
        else:
            with open(file_path, 'r', encoding='utf-16') as file:
                data = file.read()
    
        # Splitting the data into individual dictionaries
        split_data = data.split("}\n{")
    
        #the following for loop preprocess each strings so that later it can be converted into python dictionaries
        for i in range(len(split_data)):
            fragment = split_data[i].replace("\n", "") #removing end line characters
            #the following if conditions enclosed each string in second bracket so that it can be converted in python dictionaries
            if i != 0:
                fragment = "{" + fragment
            if i != len(split_data) - 1:
                fragment = fragment + "}"
                
            #there was ObjectId that, null values that were hampering to make them python dictionaries, the following lines of codes are just string replacement so that eval function works
            pattern = r'"_id" : ObjectId\("\w+"\)'  
            modified_string = re.sub(pattern, '"Year": {}'.format(file_no), fragment)
            if modified_string.find('"Box_office" : null') != -1:
                modified_string = modified_string.replace('"Box_office" : null', '"Box_office" : "null"')
            #print(modified_string)
            dict_obj = eval(modified_string)  # Convert string representation to dictionary
            list_of_dicts.append(dict_obj)
    
        # Now 'list_of_dicts' contains the dictionaries
        
    print(len(list_of_dicts))
    return list_of_dicts
    
def filter_dictionary(list_of_dicts):
    ''' input: a list of python dicts
        output: a list of python dicts
    '''
    new_list_of_dicts = []
    for i in range(len(list_of_dicts)):
        keys_to_keep = ['Year', 'imdbID', 'Genre'] # we don't need all the informations
        my_dic = list_of_dicts[i]
        
        filterd_dict = {key: my_dic[key] for key in keys_to_keep if key in my_dic}
        new_list_of_dicts.append(filterd_dict)
        
    return new_list_of_dicts
        
def create_gene_dic(df):
    '''
        input:  a dataframe
        output: one hot encoding of multiple genre labels 
    '''
    genre_set = set()

    for i in range(len(df)):
        #print(i)
        genre_list = df['Genre'][i].split(', ') #splitting string to get multiple genres

        for j in range(len(genre_list)):
            genre_set.add(genre_list[j])
            
    genre_dic = {}

    for each in genre_set:
        genre_dic[each] = [] #creating empty genre dictionary for each genres to make a dataframe


    # this following loop one hot encodes genres data
    for i in range(len(df)):
        #print(i)
        genre_list = df['Genre'][i].split(', ')
    
        for each in genre_dic.keys():
            if each in genre_list:
                genre_dic[each].append(1)
            else:
                genre_dic[each].append(0)
                
    one_hot_genre = pd.DataFrame(genre_dic)
    return one_hot_genre



def preprocess(df):
    '''
    input: a dataframe
    output: a preprocessed dataframe
    '''

   #1982 image folder was completely empty so dropping those columns
    indexes_to_drop = df.loc[df['Year'] == 1982].index
    df = df.drop(indexes_to_drop)
    df = df.reset_index(drop=True)
    
    ids = [] # that are not in folders
    # some images were not present so dropping thos metadata via following lines of codes
    for i in tqdm(range(df.shape[0])):
        img_path = 'D:\\OneDrive - University of Central Florida\\Course Work\\Fall 23\\CAP-5415-CV\\project\\movie dataset\\movie_poster_dataset_combined\\{}.jpg'.format(df['imdbID'][i])

        if os.path.exists(img_path):
            continue
        else:
            ids.append(i)
            
    df = df.drop(ids)
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset = 'imdbID', keep = 'first')
    df = df.reset_index(drop=True)
    df = df.drop(columns = ['N/A'])
    return df

if __name__ == '__main__':
    list_of_dicts = read_all_metadata_files() #reading all the metadata files
    new_list_of_dicts = filter_dictionary(list_of_dicts)
    
    genre_df = pd.DataFrame(new_list_of_dicts)
    genre_df = genre_df.dropna()
    genre_df = genre_df.reset_index(drop = True)
    #print(genre_df.info())
    label_df = create_gene_dic(genre_df)
    merged_df = pd.concat([genre_df, label_df], axis=1)
    final_df = preprocess(merged_df)
    #print(final_df['imdbID'].nunique())
    