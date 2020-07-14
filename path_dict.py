import socket
import os

class PathDict (object):
    def __init__ (self):
        hostname = socket.gethostname()
        if 'abci' in hostname:
            self.proj_root = os.path.dirname(os.path.abspath(__file__))
            self.ds_root = '/home/acb11711tx/lzq/dataset'
        else:
            self.proj_root = os.path.dirname(os.path.abspath(__file__))
            self.ds_root = '/home/lzq/dataset'
