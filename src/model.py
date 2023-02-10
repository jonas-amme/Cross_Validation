from typing import List, Tuple, Union

import torchvision
import torch
import torch.nn as nn

from torchvision.models import get_model


MODELS = [
        'resnet18',
        'resnet50',
        'vit_b_16',
        'convnext_small',
        'efficientnet_b0',
        'densenet121'
    ]
    

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
        weights: str = 'DEFAULT',
        num_classes: int = 1) -> None:
        super().__init__()

        self.model = model 
        self.weights = weights
        self.num_classes = num_classes

        self.classifier = self.build_model()


    def build_model(self):
        """Model constructor function.

        Loads a torchvision model and its pretrained weights. Then adapts the final layer
        to the binary classificaiton task.
        """
        assert self.weights in ['DEFAULT', 'IMAGENET1K_V1', 'None'], \
            'Unsupported weights for {}. Should be one of [DEFAULT, IMAGENET1K_V1, None]'.format(self.weights)
        
        weights = None if self.weights == 'None' else self.weights

        assert self.model in MODELS, \
            'Unsupported model for {}. Should be one of {}'.format(self.model, MODELS)

        # load model and/or pretrained weights
        classifier = get_model(self.model, weights=weights)

        # adapt final layer 
        if self.model in ['resnet18', 'resnet50']:
            classifier.fc = nn.Linear(classifier.fc.in_features, self.num_classes)
        elif self.model == 'vit_b_16':
            classifier.heads.head = nn.Linear(classifier.heads.head.in_features, self.num_classes)
        elif self.model == 'densenet121':
            classifier.classifier = nn.Linear(classifier.classifier.in_features, self.num_classes)
        else:
            classifier.classifier[-1] = nn.Linear(classifier.classifier[-1].in_features, self.num_classes)

        return classifier


    def forward(self, x):
        """Foward pass

        Args:
            x (Tensor): Tensor with shape [B, 3, W, H]

        Returns:
            Tuple[Tensor]: logits, probabilities and labels
        """
        logits = self.classifier(x)
        logits = logits.squeeze()
        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).float()
        return logits, Y_prob, Y_hat





class CIFAR_Classifier(Mitosis_Classifier):
    def __init__(self, model: str, weights: str = 'None', num_classes: int = 10) -> None:
        super().__init__(model, weights, num_classes)

    def forward(self, x):
        logits = self.classifier(x)
        Y_prob = torch.softmax(logits, dim=1)
        Y_hat = torch.argmax(logits, dim=1)
        return logits, Y_prob, Y_hat





        

