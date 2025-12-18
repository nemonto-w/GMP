
DATA = {
    # "path1": 'data/data_0915_0918/s1_{date}.txt',
    # "path2": 'data/data_0915_0918/s2_{date}.txt',
    "path1": 'data/data_1105_00_06/s1_{date}.txt',
    "path2": 'data/data_1105_00_06/s2_{date}.txt',
    "train_date": '1105_04',
    "test_date": '1105_05',
    "test_skip_seconds": 20,
    # "training_type": 1,  # training_type: 1,2,3 as defined in pre.py - select_data function
    # "scaler_type": 'single_scaler',  #(fit on training data set only)
    "scaler_type": 'multi_scaler', #(fit on each data set individually)
    "filter_type" : 'no_filt',
    # "filter_type" : 'filt'
}

CONSTANTS = {
    "fs": 512,
    "target_fs": 64,
    "input_seconds": 10,
    "output_seconds": 1,
    "BATCH_SIZE": 64,
    "EPOCHS": 150,
    "PATIENCE": 15,
    "NUM_FEATURES": 2,   
    "LEARNING_RATE": 1e-4
}

GRU_PARAMS = {
    "gru_units_layer1": 64,
    "dropout": 0.2,
    "gru_units_layer2": 64,
    "dense_units": 32,
}

LSTM_PARAMS = {
    "lstm_units_layer1": 64,
    "dropout": 0.2,
    "lstm_units_layer2": 64,
    "dense_units": 32,
}

'''
#install required packages
cd /path_to_the_project_folder
conda env create -f environment.yml
conda activate env_name
#remove environment or rename it
conda env remove --name isat
conda env create -f environment.yml
jupyter nbconvert --to html filename.ipynb
'''