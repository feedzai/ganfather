import torch
import torch.nn as nn


class G(nn.Module):
    def __init__(self, batch_size):
        super(G, self).__init__()

        self.batch_size = batch_size

        self.linear = nn.Sequential(
            nn.Linear(100, 400),
            nn.ReLU(True),
            nn.Linear(400, 1600),
            nn.ReLU(True),
            nn.Linear(1600, 5000),
            nn.ReLU(True),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose1d(10, 10, 4, 2, 1, bias=False),  # 10 -> 20
            nn.BatchNorm1d(10),
            nn.ReLU(True),
            nn.ConvTranspose1d(10, 10, 4, 2, 1, bias=False),  # 20 -> 40
            nn.BatchNorm1d(10),
            nn.ReLU(True),
            nn.ConvTranspose1d(10, 10, 4, 2, 1, bias=False),  # 40 -> 80
            nn.BatchNorm1d(10),
            nn.ReLU(True),
            nn.ConvTranspose1d(10, 10, 4, 2, 1, bias=False),  # 80 -> 160
            nn.BatchNorm1d(10),
            nn.ReLU(True),
        )

        self.conv_a = nn.Sequential(
            nn.ConvTranspose1d(10, 10, 4, 2, 1, bias=False),  # 160 -> 320
            nn.BatchNorm1d(10),
            nn.ReLU(True),
            nn.Conv1d(10, 1, 1, bias=False),  # 320 -> 320
            nn.Softplus(threshold=5),
        )

        self.conv_p = nn.Sequential(
            nn.ConvTranspose1d(10, 10, 4, 2, 1, bias=False),  # 160 -> 320
            nn.BatchNorm1d(10),
            nn.ReLU(True),
            nn.Conv1d(10, 1, 1, bias=False),  # 320 -> 320
            nn.Sigmoid(),
        )

        self.p_global = nn.Parameter(torch.Tensor([2 ** -6]))
        self.scaling_f = nn.Parameter(torch.Tensor([2 ** 7]))

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape((self.batch_size * 50, 10, 10))
        x = self.conv(x)
        a = self.conv_a(x) * self.scaling_f
        a = a.reshape((self.batch_size, 5, 10, 320))
        p = self.conv_p(x) * self.p_global
        p = p.reshape((self.batch_size, 5, 10, 320))
        p.clamp_(2 ** -11, 1.0)
        return a, p

    def sample(self, a, p):
        b = (torch.bernoulli(p) - p).detach() + p
        x = a * b
        return x, b


class D(nn.Module):
    def __init__(self, batch_size, WGAN=True):
        super(D, self).__init__()

        self.batch_size = batch_size

        self.conv_a = nn.Sequential(
            nn.Conv1d(1, 10, 6, 4, 1, bias=False),  # 320 -> 80
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(10, 10, 6, 4, 1, bias=False),  # 80 -> 20
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(10, 5, 6, 4, 1, bias=False),  # 20 -> 5
            nn.BatchNorm1d(5),
            nn.LeakyReLU(0.2, True),
        )

        self.conv_c = nn.Sequential(
            nn.Conv1d(1, 10, 6, 4, 1, bias=False),  # 320 -> 80
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(10, 10, 6, 4, 1, bias=False),  # 80 -> 20
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(10, 5, 6, 4, 1, bias=False),  # 20 -> 5
            nn.BatchNorm1d(5),
            nn.LeakyReLU(0.2, True),
        )

        self.agg = [
            torch.mean,
            # lambda x,y: torch.min(torch.min(x,y[1])[0],y[0])[0],
            # lambda x,y: torch.max(torch.max(x,y[1])[0],y[0])[0]
        ]

        self.linear = (
            nn.Sequential(
                nn.Linear(100 * len(self.agg), 128),
                nn.LeakyReLU(0.2, True),
                nn.Linear(128, 32),
                nn.LeakyReLU(0.2, True),
                nn.Linear(32, 1),
            )
            if WGAN
            else nn.Sequential(
                nn.Linear(100 * len(self.agg), 128),
                nn.LeakyReLU(0.2, True),
                nn.Linear(128, 32),
                nn.LeakyReLU(0.2, True),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        )

    def forward(self, a, c):
        a = a.reshape((self.batch_size * 5 * 10, 1, 320))
        a = self.conv_a(a)
        a = a.reshape((self.batch_size, 5, 2, 5, 25))

        c = c.reshape((self.batch_size * 5 * 10, 1, 320))
        c = self.conv_c(c)
        c = c.reshape((self.batch_size, 5, 2, 5, 25))

        x = torch.cat([f(x, (1, 3)) for f in self.agg for x in [a, c]], dim=1)
        x = x.reshape(self.batch_size, 100 * len(self.agg))

        x = self.linear(x)
        return x.squeeze(1)
