import numpy as np
import torch


class ImageEncoder(object):
    _instance = None
    model_repo: str
    model_name: str
    weights: str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    appearance_descriptors_model = None
    image_shape = None

    def __new__(cls, model_repo, model_name, weights):
        if not isinstance(cls._instance, cls):
            cls._instance = super(ImageEncoder, cls).__new__(cls)
            cls._instance.load_model(model_repo, model_name, weights)
        return cls._instance

    @classmethod
    def load_model(cls, model_repo, model_name, weights):
        cls.model_repo = model_repo
        cls.model_name = model_name
        cls.weights = weights
        cls.appearance_descriptors_model = torch.hub.load(
            cls.model_repo, cls.model_name, weights=cls.weights
        )
        cls.appearance_descriptors_model.fc = torch.nn.Identity()
        cls.appearance_descriptors_model = cls.appearance_descriptors_model.to(
            cls.device
        )
        cls.appearance_descriptors_model.eval()
        cls.image_shape = next(
            iter(cls.appearance_descriptors_model.parameters())
        ).shape[1:]

    @classmethod
    def update_model(cls, new_model_repo, new_model_name, new_weights):
        cls._instance.load_model(new_model_repo, new_model_name, new_weights)

    def __call__(self, image):
        image_tensor = torch.from_numpy(image).float().to(self.device)
        return self.appearance_descriptors_model(image_tensor)
