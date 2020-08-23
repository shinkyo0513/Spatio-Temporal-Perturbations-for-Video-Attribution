import socket
import os

class PathDict (object):
    def __init__ (self):
        hostname = socket.gethostname()
        if 'abci' in hostname:
            self.proj_root = os.path.dirname(os.path.abspath(__file__))
            self.ds_root = '/home/acb11711tx/lzq/dataset'
            self.epic_rltv_dir = 'epic-kitchens/frames_rgb_flow/rgb/seg_train'
        else:
            self.proj_root = os.path.dirname(os.path.abspath(__file__))
            self.ds_root = '/home/lzq/dataset'
            self.epic_rltv_dir = 'epic/seg_train'
