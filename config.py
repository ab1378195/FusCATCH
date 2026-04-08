from static import DEFAULT_HYPER_PARAMS

class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr