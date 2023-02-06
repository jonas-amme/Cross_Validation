from typing import List, Tuple, Union

import torchvision
import torch
import torch.nn as nn

from torchvision.models import get_model


class Mitosis_Classifier(nn.Module):
    """Mitosis Classifier

    Uses torchvision's pre-trained models to load a standard classifiation network.
    Adapts the final classification layer to have a single output node for 
    binary classifiction of mitosis vs. non-mitosis.

    Args:
        model (str): Model type (e.g. resnet18, resent50).
        weights (str, optional): Weight type. Defaults to 'DEFAULT'.
    """
    def __init__(
        self,
        model: str,
        weights: str = 'DEFAULT') -> None:
        super().__init__()

        assert weights in ['DEFAULT', 'IMAGENET1K_V1', None], \
            'Unsupported weights for {}. Should be one of [DEFAULT, IMAGENET1K_V1, None]'.format(weights)

        assert model in ['resnet18', 'resent50'], \
            'Unsupported model for {}. Should be one of [resnet18, resnet50]'.format(model)

        self.classifier = get_model(model, weights=weights)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, 1)


    def forward(self, x):
        """Foward pass

        Args:
            x (Tensor): Tensor with shape [B, 3, W, H]

        Returns:
            Tuple[Tensor]: logits, probabilities and labels
        """
        logits = self.classifier(x)
        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).float()
        return logits, Y_prob, Y_hat





        

