import os
import pandas as pd

class Utils:
    
    def __init__(self, name):
        self._name = name

    @classmethod
    def reportGPU(cls, tf):

        if tf.test.gpu_device_name():
            print('GPU found: {}'.format(tf.test.gpu_device_name()))
        else:
            print("No GPU found")

    @classmethod
    def makePath(cls, path, mode):

        try:
            original_umask = os.umask(0)
            if not os.path.exists(path):
                os.makedirs(path, mode=mode)
        finally:
            os.umask(original_umask)

    @classmethod
    def printPandasGroupBy(cls, list, columns, category):

        df = pd.DataFrame(list)
        df.columns = columns
        print(df.groupby(category).count())