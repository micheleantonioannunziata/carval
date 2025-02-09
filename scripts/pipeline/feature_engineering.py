class feature_engineering:
    def __init__(self):
        self.anno_rif = 2015

    def transform(self, data):
        data['età'] = self.anno_rif - data['anno produzione']
        data.drop(columns=['anno produzione'], inplace=True)
        return data
