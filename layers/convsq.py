from torch.nn import Sequential, BatchNorm1d, ReLU, Tanh, Conv1d


class ConvSq(Sequential):
    # 初始化方法，定义卷积块的参数和层结构
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        norm_layer=BatchNorm1d,
        activation_layer=ReLU,
        dilation=1,
        inplace=True,
        bias=None,
        conv_layer=Conv1d,
        first_conv=True,
    ):
        if first_conv:
            norm_channels = out_channels
        else:
            norm_channels = in_channels

        if bias is None:
            bias = norm_layer is None
        layers = []

        if norm_layer is not None:
            layers.append(norm_layer(norm_channels))

        if activation_layer is not None:
            if activation_layer == Tanh:
                layers.append(activation_layer())
            else:
                layers.append(activation_layer(inplace=inplace))

        if first_conv:
            layers.insert(
                0,
                conv_layer(
                    in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias
                ),
            )
        else:
            layers.append(
                conv_layer(
                    in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias
                )
            )
        super(ConvSq, self).__init__(*layers)
        self.out_channels = out_channels
