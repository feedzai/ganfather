import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.scale = (
            None
            if input_size == output_size
            else torch.nn.Linear(input_size, output_size, False)
        )
        self.l1 = torch.nn.Linear(input_size, output_size, False)
        self.bn1 = torch.nn.BatchNorm1d(output_size)
        self.l2 = torch.nn.Linear(output_size, output_size, False)
        self.bn2 = torch.nn.BatchNorm1d(output_size)
        self.relu = torch.nn.ReLU(True)

    def forward(self, x):
        identity = self.scale(x) if (self.scale is not None) else x
        out = self.relu(self.bn1(self.l1(x)))
        out = self.bn2(self.l2(out))
        out += identity
        return self.relu(out)


class G(torch.nn.Module):
    def __init__(
        self,
        common_l=[128, 128, 256, 256, 512, 512],
        separate_l=[512, 256, 128, 64],
        layer=ResidualBlock,
    ):
        super(G, self).__init__()

        self.common = [
            layer(common_l[i], common_l[i + 1]) for i in range(len(common_l) - 1)
        ]
        self.common = torch.nn.Sequential(*self.common)

        self.separate_a = [
            layer(separate_l[i], separate_l[i + 1]) for i in range(len(separate_l) - 1)
        ] + [torch.nn.Linear(separate_l[-1], 3706), torch.nn.Sigmoid()]
        self.separate_a = torch.nn.Sequential(*self.separate_a)

        self.separate_p = [
            layer(separate_l[i], separate_l[i + 1]) for i in range(len(separate_l) - 1)
        ] + [torch.nn.Linear(separate_l[-1], 3706), torch.nn.Sigmoid()]
        self.separate_p = torch.nn.Sequential(*self.separate_p)

        return

    def forward(self, x):
        x = self.common(x)
        a = self.separate_a(x) * 4 + 1
        p = self.separate_p(x)
        return a, p

    def sample(self, a, p):
        b = (torch.bernoulli(p) - p).detach() + p
        return a * b, b


class D(torch.nn.Module):
    def __init__(
        self,
        separate_l=[64, 128, 256, 256],
        common_l=[512, 512, 256, 256, 128, 128],
        layer=ResidualBlock,
        WGAN=True,
    ):
        super(D, self).__init__()

        self.separate_a = [torch.nn.Linear(3706, separate_l[0])] + [
            layer(separate_l[i], separate_l[i + 1]) for i in range(len(separate_l) - 1)
        ]
        self.separate_a = torch.nn.Sequential(*self.separate_a)

        self.separate_c = [torch.nn.Linear(3706, separate_l[0])] + [
            layer(separate_l[i], separate_l[i + 1]) for i in range(len(separate_l) - 1)
        ]
        self.separate_c = torch.nn.Sequential(*self.separate_c)

        self.common = [
            layer(common_l[i], common_l[i + 1]) for i in range(len(common_l) - 1)
        ]
        if WGAN:
            self.common += [torch.nn.Linear(common_l[-1], 1)]
        else:
            self.common += [torch.nn.Linear(common_l[-1], 1), torch.nn.Sigmoid()]
        self.common = torch.nn.Sequential(*self.common)

    def forward(self, a, c):
        a = self.separate_a(a)
        c = self.separate_c(c)
        x = torch.cat([a, c], dim=1)
        x = self.common(x)
        return x
