import torch.nn as nn
import LibMTL.model.resnet as resnet

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # 使用预定义的 ResNet，但不包含 AvgPool 和 FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.feature_dim = orig_resnet.feature_dim

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # 带步长的卷积
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # 其他卷积
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def resnet_dilated(basenet, pretrained=True, dilate_scale=8):
    r""`Dilated Residual Networks <https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Dilated_Residual_Networks_CVPR_2017_paper.pdf>`_ 模型。

    Args:
        basenet (str): ResNet 的类型。
        pretrained (bool): 如果为 True，返回在 ImageNet 上预训练的模型。
        dilate_scale ({8, 16}, default=8): 膨胀过程的类型。
    """
    return ResnetDilated(resnet.__dict__[basenet](pretrained=pretrained), dilate_scale=dilate_scale)
