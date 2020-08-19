from pathlib import Path

# run as soon as imported
home = Path(__file__).resolve().parents[1]

log_path = home / 'logs'
data_path = home / 'data'
model_path = home / 'models'
src_path = home / 'src'

# data_path = home/'data'/'raw'
# data_interim_path = home/'data'/'interim'
# data_train = data_path/'jpeg'/'train'
# data_test = data_path/'jpeg'/'test'

data_raw_path = data_path/'raw'
data_interim_path = data_path/'interim'
data_train = data_raw_path/'jpeg'/'train'
data_test = data_raw_path/'jpeg'/'test'

interim_pred_path = data_interim_path/'interim_predictions'
interim_path_test = interim_pred_path/'test'
interim_path_train = interim_pred_path/'train'
# raw_path = Path('../data/raw')