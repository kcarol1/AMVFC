class RecordResult:
    def __init__(self):
        self.best_result_record = None
        self.metric_name = None

    def update(self, result: dict, metric: str):
        if self.best_result_record is None or result[metric] > self.best_result_record[metric]:
            self.best_result_record = result
            self.metric_name = metric

    def get_best(self):
        return self.best_result_record

import scipy.io as sio
class Mat_Reader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.keys = []
        self.mat = None
    def read_mat(self):
        self.mat = sio.loadmat(self.file_path)
        self.keys = [key for key in self.mat.keys() if not key.startswith('__')]
        print(self.keys)
        
    def get_data(self, key=None):
        if key is None:
            key = self.keys[0]
        data = self.mat[key]
        return data