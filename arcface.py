import math
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import CFG
import albumentations
from albumentations.pytorch.transforms import ToTensorV2


class ArcFaceModule(nn.Module):
    def __init__(self, in_features, out_features, scale, margin, easy_margin=False, ls_eps=0.0):
        super(ArcFaceModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):

        # cosine = X.W = ||X|| .||W|| . cos(theta)
        # if X and W are normalize then dot product X, W = will be cos theta
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # phi = cos(theta + margin) = cos theta . cos(margin) -  sine theta .  sin(margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=CFG.device)
        # one hot encoded
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        #  output = label == True ? phi : cosine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # scale the output
        output *= self.scale
        # return cross entropy loss on scalled output
        return output, nn.CrossEntropyLoss()(output, label)


class ShopeeEncoderBackBone(nn.Module):

    def __init__(self,
                 model_name='tf_efficientnet_b3',
                 loss_fn='ArcFace',
                 classes=CFG.classes,
                 fc_dim=CFG.fc_dim,
                 pretrained=False,
                 use_fc=True,
                 isTraining=True
                 ):

        super(ShopeeEncoderBackBone, self).__init__()

        # create bottlenack backbone network from pretrained model
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc
        self.loss_fn = loss_fn
        self.isTraining = isTraining

        # build top fc layers (Embedding that we are looking at testing time to represent the entire image)
        # this will work as regularizer
        if self.use_fc:
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.init_params()
            in_features = fc_dim
        self.loss_fn = loss_fn
        if self.loss_fn == 'softmax':
            self.final = nn.Linear(in_features, CFG.classes)
        elif self.loss_fn == 'ArcFace':
            self.final = ArcFaceModule(in_features,
                                       CFG.classes,
                                       scale=30,
                                       margin=0.5,
                                       easy_margin=False,
                                       ls_eps=0.0)

    def forward(self, image, label):
        features = self.get_features(image)
        if self.isTraining:
            logits = self.final(features, label)
            return logits
        else:
            return features

    def init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def get_features(self, inp):
        batch_dim = inp.shape[0]
        inp = self.backbone(inp)
        inp = self.pooling(inp).view(batch_dim, -1)
        if self.use_fc and self.isTraining:
            inp = self.dropout(inp)
            inp = self.fc(inp)
            inp = self.bn(inp)

        return inp


def get_test_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.img_size, CFG.img_size, always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )


def getAugmentation(IMG_SIZE=CFG.img_size, isTraining=CFG.isTraining):
    if isTraining:
        return albumentations.Compose([
            albumentations.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.75),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(p=1.0)
        ])
    else:
        return albumentations.Compose(
            [
                albumentations.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
                albumentations.Normalize(),
                ToTensorV2(p=1.0)
            ]
        )