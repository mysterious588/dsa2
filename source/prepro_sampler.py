# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
https://github.com/Automunge/AutoMunge#library-of-transformations
Library of Transformations
Library of Transformations Subheadings:
Intro
Numerical Set Normalizations
Numerical Set Transformations
Numercial Set Bins and Grainings
Sequential Numerical Set Transformations
Categorical Set Encodings
Date-Time Data Normalizations
Date-Time Data Bins
Differential Privacy Noise Injections
Misc. Functions
String Parsing
More Efficient String Parsing
Multi-tier String Parsing
List of Root Categories
List of Suffix Appenders
Other Reserved Strings
Root Category Family Tree Definitions
"""
import warnings
warnings.filterwarnings('ignore')
import sys, gc, os, pandas as pd, json, copy
import numpy as np

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


#### Debuging state (Ture/False)
DEBUG_=True

####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)

def logs(*s):
    if DEBUG_:
        print(*s, flush=True)


def log_pd(df, *s, n=0, m=1):
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump,  df.head(n), flush=True)


from util_feature import  save, load_function_uri, load
import util_feature
####################################################################################################
####################################################################################################
def save_features(df, name, path):
    """
    :param df:
    :param name:
    :param path:
    :return:
    """
    if path is not None :
       os.makedirs( f"{path}/{name}" , exist_ok=True)
       if isinstance(df, pd.Series):
           df0=df.to_frame()
       else:
           df0=df
       df0.to_parquet( f"{path}/{name}/features.parquet")




###################################################################################################
##### Filtering / cleaning rows :   #########################################################
def pd_filter_rows(df, col, pars):
    import re
    coly = col
    filter_pars =  pars
    def isfloat(x):
        #x = re.sub("[!@,#$+%*:()'-]", "", str(x))
        try :
            a= float(x)
            return 1
        except:
            return 0

    ymin, ymax = pars.get('ymin', -9999999999.0), filter_pars.get('ymax', 999999999.0)

    df['_isfloat'] = df[ coly ].apply(lambda x : isfloat(x),axis=1 )
    df = df[ df['_isfloat'] > 0 ]
    df = df[df[coly] > ymin]
    df = df[df[coly] < ymax]
    del df['_isfloat']

    return df, col


def pd_filter_resample(df=None, col=None, pars=None):
    """
        Over-sample, Under-sample
    """
    prefix = 'col_imbalance'
    ######################################################################################
    from imblearn.over_sampling import SMOTE

    model_resample = { 'SMOTE' : SMOTE}[  pars.get("model_name", 'SMOTE') ]

    pars_resample  = pars.get('pars_resample',
                             {'sampling_strategy' : 'auto', 'random_state':0, 'k_neighbors':5, 'n_jobs': 2})

    if 'path_pipeline' in pars :   #### Inference time
        return df, {'col_new': col }
        #gp   = load(pars['path_pipeline'] + f"/{prefix}_model.pkl" )
        #pars = load(pars['path_pipeline'] + f"/{prefix}_pars.pkl" )

    else :     ### Training time
        colX          = col # [col_ for col_ in col if col_ not in coly]
        train_X       = df[colX].fillna(method='ffill')
        coly     = pars['coly']
        train_y  = pars['dfy']
        gp       = model_resample( **pars_resample)
        X_resample, y_resample = gp.fit_resample(train_X, train_y)

        df2       = pd.DataFrame(X_resample, columns = col, index=train_X.index)
        df2[coly] = y_resample


    col_new = col
    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(df2, 'df_resample', pars['path_features_store'])
       save(gp,             pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(col,            pars['path_pipeline_export'] + f"/{prefix}.pkl" )
       save(pars_resample,   pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )


    col_pars = {'prefix' : prefix , 'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
       prefix :  col_new  ### list
    }
    return df2, col_pars





##### Label processing   ##################################################################

def pd_autoencoder(df, col, pars):
    """"
    (4) Autoencoder
An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore noise.
(i) Feed Forward
The simplest form of an autoencoder is a feedforward, non-recurrent neural network similar to single layer perceptrons that participate in multilayer perceptrons
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
import numpy as np
def encoder_dataset(df, drop=None, dimesions=20):
  if drop:
    train_scaled = minmax_scale(df.drop(drop,axis=1).values, axis = 0)
  else:
    train_scaled = minmax_scale(df.values, axis = 0)
  # define the number of encoding dimensions
  encoding_dim = dimesions
  # define the number of features
  ncol = train_scaled.shape[1]
  input_dim = tf.keras.Input(shape = (ncol, ))
  # Encoder Layers
  encoded1 = tf.keras.layers.Dense(3000, activation = 'relu')(input_dim)
  encoded2 = tf.keras.layers.Dense(2750, activation = 'relu')(encoded1)
  encoded3 = tf.keras.layers.Dense(2500, activation = 'relu')(encoded2)
  encoded4 = tf.keras.layers.Dense(750, activation = 'relu')(encoded3)
  encoded5 = tf.keras.layers.Dense(500, activation = 'relu')(encoded4)
  encoded6 = tf.keras.layers.Dense(250, activation = 'relu')(encoded5)
  encoded7 = tf.keras.layers.Dense(encoding_dim, activation = 'relu')(encoded6)
  encoder = tf.keras.Model(inputs = input_dim, outputs = encoded7)
  encoded_input = tf.keras.Input(shape = (encoding_dim, ))
  encoded_train = pd.DataFrame(encoder.predict(train_scaled),index=df.index)
  encoded_train = encoded_train.add_prefix('encoded_')
  if drop:
    encoded_train = pd.concat((df[drop],encoded_train),axis=1)
  return encoded_train
df_out = mapper.encoder_dataset(df.copy(), ["Close_1"], 15); df_out.head()
    """
    pass


def pd_colcat_encoder_generic(df, col, pars):
    """
        Create a Class or decorator
        https://pypi.org/project/category-encoders/
        encoder = ce.BackwardDifferenceEncoder(cols=[...])
        encoder = ce.BaseNEncoder(cols=[...])
        encoder = ce.BinaryEncoder(cols=[...])
        encoder = ce.CatBoostEncoder(cols=[...])
        encoder = ce.CountEncoder(cols=[...])
        encoder = ce.GLMMEncoder(cols=[...])
        encoder = ce.HashingEncoder(cols=[...])
        encoder = ce.HelmertEncoder(cols=[...])
        encoder = ce.JamesSteinEncoder(cols=[...])
        encoder = ce.LeaveOneOutEncoder(cols=[...])
        encoder = ce.MEstimateEncoder(cols=[...])
        encoder = ce.OneHotEncoder(cols=[...])
        encoder = ce.OrdinalEncoder(cols=[...])
        encoder = ce.SumEncoder(cols=[...])
        encoder = ce.PolynomialEncoder(cols=[...])
        encoder = ce.TargetEncoder(cols=[...])
        encoder = ce.WOEEncoder(cols=[...])
    """
    prefix     = "colcat_encoder_generic"
    pars_model = None
    if 'path_pipeline' in  pars  :   ### Load during Inference
       colcat_encoder = load( pars['path_pipeline'] + f"/{prefix}.pkl" )
       pars_model     = load( pars['path_pipeline'] + f"/{prefix}_pars.pkl" )
       #model         = load( pars['path_pipeline'] + f"/{prefix}_model.pkl" )

    ####### Custom Code ###############################################################
    from category_encoders import HashingEncoder, WOEEncoder
    pars_model         = pars.get('model_pars', {})  if pars_model is None else pars_model
    pars_model['cols'] = col
    model_name         = pars.get('model_name', 'HashingEncoder')

    model_class        = { 'HashingEncoder' : HashingEncoder  }[model_name]
    model              = model_class(**pars_model)
    dfcat_encoder      = model.fit_transform(df[col])

    dfcat_encoder.columns = [t + "_cod" for t in dfcat_encoder.columns ]
    colcat_encoder        = list(dfcat_encoder.columns)


    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(dfcat_encoder, 'dfcat_encoder', pars['path_features_store'])
       save(model,           pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(pars_model,      pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )
       save(colcat_encoder,  pars['path_pipeline_export'] + f"/{prefix}.pkl" )

    col_pars = { 'prefix' : prefix,  'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
     'colcat_encoder_generic' :  colcat_encoder  ### list
    }
    return dfcat_encoder, col_pars



def os_convert_topython_code(txt):
    # from sympy import sympify
    # converter = {
    #     'sub': lambda x, y: x - y,
    #     'div': lambda x, y: x / y,
    #     'mul': lambda x, y: x * y,
    #     'add': lambda x, y: x + y,
    #     'neg': lambda x: -x,
    #     'pow': lambda x, y: x ** y
    # }
    # formula = sympify( txt, locals=converter)
    # print(formula)
    pass


def save_json(js, pfile, mode='a'):
    import  json
    with open(pfile, mode=mode) as fp :
        json.dump(js, fp)


def pd_col_genetic_transform(df=None, col=None, pars=None):
    """
        Find Symbolic formulae for faeture engineering
    """
    prefix = 'col_genetic'
    ######################################################################################
    from gplearn.genetic import SymbolicTransformer
    from gplearn.functions import make_function
    import random

    colX          = col # [col_ for col_ in col if col_ not in coly]
    train_X       = df[colX].fillna(method='ffill')
    feature_name_ = colX

    def squaree(x):  return x * x
    square_ = make_function(function=squaree, name='square_', arity=1)

    function_set = pars.get('function_set',
                            ['add', 'sub', 'mul', 'div',  'sqrt', 'log', 'abs', 'neg', 'inv','tan', square_])
    pars_genetic = pars.get('pars_genetic',
                             {'generations': 5, 'population_size': 10,  ### Higher than nb_features
                              'metric': 'spearman',
                              'tournament_size': 20, 'stopping_criteria': 1.0, 'const_range': (-1., 1.),
                              'p_crossover': 0.9, 'p_subtree_mutation': 0.01, 'p_hoist_mutation': 0.01,
                              'p_point_mutation': 0.01, 'p_point_replace': 0.05,
                              'parsimony_coefficient' : 0.005,   ####   0.00005 Control Complexity
                              'max_samples' : 0.9, 'verbose' : 1,

                              #'n_components'      ### Control number of outtput features  : n_components
                              'random_state' :0, 'n_jobs' : 4,
                              })

    if 'path_pipeline' in pars :   #### Inference time
        gp   = load(pars['path_pipeline'] + f"/{prefix}_model.pkl" )
        pars = load(pars['path_pipeline'] + f"/{prefix}_pars.pkl" )

    else :     ### Training time
        coly     = pars['coly']
        train_y  = pars['dfy']
        gp = SymbolicTransformer(hall_of_fame  = train_X.shape[1] + 1,  ### Buggy
                                 n_components  = pars_genetic.get('n_components', train_X.shape[1] ),
                                 feature_names = feature_name_,
                                 function_set  = function_set,
                                 **pars_genetic)
        gp.fit(train_X, train_y)

    ##### Transform Data  #########################################
    df_genetic = gp.transform(train_X)
    tag = random.randint(0,10)   #### UNIQUE TAG
    col_genetic  = [ f"gen_{tag}_{i}" for i in range(df_genetic.shape[1])]
    df_genetic   = pd.DataFrame(df_genetic, columns= col_genetic, index = train_X.index )
    df_genetic.index = train_X.index
    pars_gen_all = {'pars_genetic'  : pars_genetic , 'function_set' : function_set }

    ##### Formulae Exrraction #####################################
    formula   = str(gp).replace("[","").replace("]","")
    flist     = formula.split(",\n")
    form_dict = {  x: flist[i]  for i,x in enumerate(col_genetic) }
    pars_gen_all['formulae_dict'] = form_dict
    log("########## Formulae ", form_dict)
    # col_pars['map_dict'] = dict(zip(train_X.columns.to_list(), feature_name_))

    col_new = col_genetic

    ###################################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(df_genetic, 'df_genetic', pars['path_features_store'])
       save(gp,             pars['path_pipeline_export'] + f"/{prefix}_model.pkl" )
       save(col_genetic,    pars['path_pipeline_export'] + f"/{prefix}.pkl" )
       save(pars_gen_all,   pars['path_pipeline_export'] + f"/{prefix}_pars.pkl" )
       # save(form_dict,      pars['path_pipeline_export'] + f"/{prefix}_formula.pkl")
       save_json(form_dict, pars['path_pipeline_export'] + f"/{prefix}_formula.json")   ### Human readable


    col_pars = {'prefix' : prefix , 'path' :   pars.get('path_pipeline_export', pars.get('path_pipeline', None)) }
    col_pars['cols_new'] = {
       prefix :  col_new  ### list
    }
    return df_genetic, col_pars





def pd_augmentation_sdv(df, col=None, pars={})  :
    '''
    Using SDV Variation Autoencoders, the function augments more data into the dataset
    params:
            df          : (pandas dataframe) original dataframe
            col : column name for data enancement
            pars        : (dict - optional) contains:                                          
                n_samples     : (int - optional) number of samples you would like to add, defaul is 10%
                primary_key   : (String - optional) the primary key of dataframe
                aggregate  : (boolean - optional) if False, prints SVD metrics, else it averages them
                path_model_save: saving location if save_model is set to True
                path_model_load: saved model location to skip training
                path_data_new  : new data where saved
    returns:
            df_new      : (pandas dataframe) df with more augmented data
            col         : (list of strings) same columns
    '''
    n_samples       = pars.get('n_samples', max(1, int(len(df) * 0.10) ) )   ## Add 10% or 1 sample by default value
    primary_key     = pars.get('colid', None)  ### Custom can be created on the fly
    metrics_type    = pars.get('aggregate', False)
    path_model_save = pars.get('path_model_save', 'data/output/ztmp/')
    model_name      = pars.get('model_name', "TVAE")
    
    # importing libraries
    try:
        #from sdv.demo import load_tabular_demo
        from sdv.tabular import TVAE
        from sdv.tabular import CTGAN
        from sdv.timeseries import PAR
        from sdv.evaluation import evaluate
        import ctgan
        
        if ctgan.__version__ != '0.3.1.dev0':
            raise Exception('ctgan outdated, updating...')
    except:
        os.system("pip install sdv")
        os.system('pip install ctgan==0.3.1.dev0')
        from sdv.tabular import TVAE
        from sdv.tabular import CTGAN
        from sdv.timeseries import PAR
        from sdv.evaluation import evaluate      
    

    # model fitting 
    if 'path_model_load' in pars:
            model = load(pars['path_model_load'])
    else:
            log('##### Training Started #####')
                
            model = {'TVAE' : TVAE, 'CTGAN' : CTGAN, 'PAR' : PAR}[model_name]
            if model_name == 'PAR':
                model = model(entity_columns = pars['entity_columns'],
                              context_columns = pars['context_columns'],
                              sequence_index = pars['sequence_index'])
            else:
                model = model(primary_key=primary_key)   
            model.fit(df)
            log('##### Training Finshed #####')
            try:
                 save(model, path_model_save )
                 log('model saved at: ', path_model_save  )
            except:
                 log('saving model failed: ', path_model_save)

    log('##### Generating Samples #############')
    new_data = model.sample(n_samples)
    log_pd( new_data, n=7)
    
   
    log('######### Evaluation Results #########')
    if metrics_type == True:
      evals = evaluate(new_data, df, aggregate= True )        
      log(evals)
    else:
      evals = evaluate(new_data, df, aggregate= False )        
      log_pd(evals, n=7)
    
    # appending new data    
    df_new = df.append(new_data)
    log(str(len(df_new) - len(df)) + ' new data added')
    
    if 'path_newdata' in pars :
        new_data.to_parquet( pars['path_newdata'] + '/features.parquet' ) 
        log('###### df augmentation save on disk', pars['path_newdata'] )    
    
    log('###### augmentation complete ######')
    return df_new, col


def test_sdv_vae():
    from sklearn.datasets import load_boston
    
    # loading boston data
    data = load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    log_pd(df)    
    
    log('##### testing augmentation #####')    
    path = os.getcwd() + '\zz_model_vae_augmentation.pkl'
    pars = {'path_model_save': path}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)
    
    log('####### Generating using saved model test started #######')    
    pars = {'path_model_load': path}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)
    

def test_sdv_ctgan():
    from sklearn.datasets import load_boston
    
    # loading boston data
    data = load_boston()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    log_pd(df)    
    
    log('##### testing augmentation #####')    
    path = os.getcwd() + '\zz_model_ctgan_augmentation.pkl'
    pars = {'path_model_save': path,
            'model_name': 'CTGAN'}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)
    
    log('####### Generating using saved model test started #######')    
    pars = {'path_model_load': path}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)
    
    
def test_sdv_par():
    try:
        from sdv.demo import load_timeseries_demo
    except:
        os.system('pip install sdv')
        from sdv.demo import load_timeseries_demo
        
    df = load_timeseries_demo()
    log_pd(df)
    entity_columns = ['Symbol']
    context_columns = ['MarketCap', 'Sector', 'Industry']
    sequence_index = 'Date'
    
    log('##### testing augmentation #####')
    path = os.getcwd() + '\zz_model_par_augmentation.pkl'
    pars = {'path_model_save': path,
            'model_name': 'PAR',
            'entity_columns' : entity_columns,
            'context_columns': context_columns,
            'sequence_index' : sequence_index,
            'n_samples' : 5}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)
    
    log('####### Generating using saved model test started #######')    
    pars = {'path_model_load': path,
            'n_samples' : 5}
    df_new, _ = pd_augmentation_sdv(df, pars=pars)
    
    
def pd_col_covariate_shift_adjustment():
   """
    https://towardsdatascience.com/understanding-dataset-shift-f2a5a262a766
     Covariate shift has been extensively studied in the literature, and a number of proposals to work under it have been published. Some of the most important ones include:
        Weighting the log-likelihood function (Shimodaira, 2000)
        Importance weighted cross-validation (Sugiyama et al, 2007 JMLR)
        Integrated optimization problem. Discriminative learning. (Bickel et al, 2009 JMRL)
        Kernel mean matching (Gretton et al., 2009)
        Adversarial search (Globerson et al, 2009)
        Frank-Wolfe algorithm (Wen et al., 2015)
import numpy as np
from scipy import sparse
# .. for plotting ..
import pylab as plt
# .. to generate a synthetic dataset ..
from sklearn import datasets
n_samples, n_features = 1000, 10000
A, b = datasets.make_regression(n_samples, n_features)
def FW(alpha, max_iter=200, tol=1e-8):
    # .. initial estimate, could be any feasible point ..
    x_t = sparse.dok_matrix((n_features, 1))
    trace = []  # to keep track of the gap
    # .. some quantities can be precomputed ..
    Atb = A.T.dot(b)
    for it in range(max_iter):
        # .. compute gradient. Slightly more involved than usual because ..
        # .. of the use of sparse matrices ..
        Ax = x_t.T.dot(A.T).ravel()
        grad = (A.T.dot(Ax) - Atb)
        # .. the LMO results in a vector that is zero everywhere except for ..
        # .. a single index. Of this vector we only store its index and magnitude ..
        idx_oracle = np.argmax(np.abs(grad))
        mag_oracle = alpha * np.sign(-grad[idx_oracle])
        g_t = x_t.T.dot(grad).ravel() - grad[idx_oracle] * mag_oracle
        trace.append(g_t)
        if g_t <= tol:
            break
        q_t = A[:, idx_oracle] * mag_oracle - Ax
        step_size = min(q_t.dot(b - Ax) / q_t.dot(q_t), 1.)
        x_t = (1. - step_size) * x_t
        x_t[idx_oracle] = x_t[idx_oracle] + step_size * mag_oracle
    return x_t, np.array(trace)
# .. plot evolution of FW gap ..
sol, trace = FW(.5 * n_features)
plt.plot(trace)
plt.yscale('log')
plt.xlabel('Number of iterations')
plt.ylabel('FW gap')
plt.title('FW on a Lasso problem')
plt.grid()
plt.show()
sparsity = np.mean(sol.toarray().ravel() != 0)
print('Sparsity of solution: %s%%' % (sparsity * 100))
   """
   pass

if __name__ == "__main__":
    import fire
    fire.Fire()
    
    
    
    
    
    
    
    
"""


def pd_generic_transform(df, col=None, pars={}, model=None)  :
 
     Transform or Samples using  model.fit()   model.sample()  or model.transform()
    params:
            df    : (pandas dataframe) original dataframe
            col   : column name for data enancement
            pars  : (dict - optional) contains:                                          
                path_model_save: saving location if save_model is set to True
                path_model_load: saved model location to skip training
                path_data_new  : new data where saved 
    returns:
            model, df_new, col, pars
   
    path_model_save = pars.get('path_model_save', 'data/output/ztmp/')
    pars_model      = pars.get('pars_model', {} )
    model_method    = pars.get('method', 'transform')
    
    # model fitting 
    if 'path_model_load' in pars:
            model = load(pars['path_model_load'])
    else:
            log('##### Training Started #####')
            model = model( **pars_model)
            model.fit(df)
            log('##### Training Finshed #####')
            try:
                 save(model, path_model_save )
                 log('model saved at: ' + path_model_save  )
            except:
                 log('saving model failed: ', path_model_save)

    log('##### Generating Samples/transform #############')    
    if model_method == 'sample' :
        n_samples =pars.get('n_samples', max(1, 0.10 * len(df) ) )
        new_data  = model.sample(n_samples)
        
    elif model_method == 'transform' :
        new_data = model.transform(df.values)
    else :
        raise Exception("Unknown", model_method)
        
    log_pd( new_data, n=7)    
    if 'path_newdata' in pars :
        new_data.to_parquet( pars['path_newdata'] + '/features.parquet' ) 
        log('###### df transform save on disk', pars['path_newdata'] )    
    
    return model, df_new, col, pars



"""
    
    
    
