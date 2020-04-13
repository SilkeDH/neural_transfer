import neural_transfer.config as cfg
import subprocess
from os import path

def download_model(name):
    try:
        nums = [cfg.MODELS_DIR, name]
        model_path = '{0}/{1}.pt'.format(*nums)
        cat_path = '{0}/{1}.pkl'.format(*nums)
        
        if not path.exists(model_path) or not path.exists(cat_path):
            remote_nums = [cfg.REMOTE_MODELS_DIR, name]
            remote_model_path = '{0}/{1}.pt'.format(*remote_nums)
            remote_cat_path = '{0}/{1}.pkl'.format(*remote_nums)
            print('[INFO] Model not found, downloading model...')
            # from "rshare" remote storage into the container
            command = (['rclone', 'copy', '--progress', remote_model_path, cfg.MODELS_DIR])
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = result.communicate()
            command = (['rclone', 'copy', '--progress', remote_cat_path, cfg.MODELS_DIR])
            result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = result.communicate()
            print('[INFO] Finished.')
        else:
            print("[INFO] Model found.")
            
    except OSError as e:
        output, error = None, e