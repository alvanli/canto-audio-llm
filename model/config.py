from transformers import PretrainedConfig

class DiVAConfig(PretrainedConfig):
    model_type = "diva"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
