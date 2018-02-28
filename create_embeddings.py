# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:55:58 2017

@author: darsh
"""

def calcVectors(data , y):
    import pandas as pd
    import numpy as np
    import utils
    import math
    
    data = data
    y = y
    categorical_vars = data.select_dtypes(['category']).columns
    varsDict= []
    for i, var in enumerate(categorical_vars):
        varsDict.append({u:i for i, u in enumerate(data[var].unique())})
    for i, var in enumerate(categorical_vars):
        data[var] = data[var].apply(lambda x: varsDict[i][x])
        counts = []
    for i, var in enumerate(categorical_vars):
        counts.append(data[var].nunique())
        emblayers = []

    for i, var in enumerate(categorical_vars):
        emblayers.append(embedding_input(var , counts[i] , np.round(math.sqrt(counts[i]) + 1).astype('int') , 1e-5))
        
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_vars = data.select_dtypes(include=numerics).columns
    pred = numerical_vars[pd.Series(numerical_vars).isin([y])]
    
    input_vars = numerical_vars[~pd.Series(numerical_vars).isin([y])]
    emb_layers_numeric = []
    for i , var in enumerate(input_vars):
        emb_layers_numeric.append([Input(shape = (1,1) , dtype = 'float32' , name = input_vars[i])])
        concat_layers =[]
    for i , layer in enumerate(emblayers):
        concat_layers.append(emblayers[i][1])
        
    for i , layer in enumerate(emb_layers_numeric):
        concat_layers.append(emb_layers_numeric[0][0])
        
    inp = merge(concat_layers , mode = 'concat')
    from keras.layers import PReLU
    
    inp1 = Flatten()(inp)
    inp1 = Dropout(0.1)(inp1)
    x = Dense(400  , name = 'L1')(inp1)
    x = PReLU(name = 'P1')(x)
    x = BatchNormalization(name = 'BN1')(x)
    inp2 = Dropout(0.1)(x)
    
    x = Dense(300, name = 'L2')(inp2)
    x = PReLU(name = 'P2')(x)
    x = BatchNormalization(name = 'BN2')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(200 , name = 'L3')(x)
    x = PReLU(name = 'P3')(x)
    x = BatchNormalization(name = 'BN3')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(100 , name = 'L4')(x)
    x = PReLU(name = 'P4')(x)
    x = BatchNormalization(name = 'BN4')(x)
    x = Dropout(0.2)(x)
    
    m = Dense(100 , name = 'm1')(inp2)
    m = PReLU(name = 'mP1')(m)
    m = BatchNormalization(name = 'mBN1')(x)
    m = Dropout(0.2)(m)
    
    x = merge([x, m] , mode = 'sum')
    
    x = Dense(50 , name = 'L5')(x)
    x = PReLU(name = 'P5')(x)
    x = Dropout(0.3)(x)
    x = Dense(1 , name = 'L6')(x)
    outputs = PReLU(name = 'P6')(x)
    
    concat_layers_inp = []
    for i , layer in enumerate(emblayers):
        concat_layers_inp.append(emblayers[i][0])
        
    for i , layer in enumerate(emb_layers_numeric):
        concat_layers_inp.append(emb_layers_numeric[i][0])
    
    nn = Model(inputs = concat_layers_inp, outputs = outputs)
    
    optimizer = Adam()
    nn.compile(optimizer=optimizer , loss = 'mse')
    print(nn.summary())
    
    input_data_list = []
    for i , var in enumerate(categorical_vars):
        input_data_list.append(np.array(data[var]))
    
    for i , var in enumerate(input_vars):
        input_data_list.append(np.array(data[var][: , np.newaxis , np.newaxis]))
        
    op = np.array(data[y])
    
    nn.optimizer.lr = 1e-6
    nn.fit(input_data_list , op , batch_size = 2048 , 
            epochs = 3, validation_split = 0.1 , shuffle = True)
    
    nn.optimizer.lr = 1e-3
    
    nn.fit(input_data_list , op , batch_size = 2048 , 
            epochs = 20, validation_split = 0.1 , shuffle = True)
    
    nn.optimizer.lr = 1e-5
    nn.fit(input_data_list , op , batch_size = 2048 , 
            epochs = 5, validation_split = 0.1 , shuffle = True)
    
    export_layers = nn.layers[len(categorical_vars):(2*len(categorical_vars))]
    
    embeddings_export_list = []
    for i , layer in enumerate(export_layers):
        embeddings_export_list.append([layer.get_weights()[0][0]])
        
    return(embeddings_export_list)
    
def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)
