import torch


class Proxy(torch.nn.Module):
    def __init__(self, shape, and_type):
        super(Proxy, self).__init__()
        # M = bs*M
        self.M, self.S, self.D, self.T = shape
        self.and_type = and_type

    def forward(self, a, c):

        # This method was removed because
        # the rules' logic is confidential

        return 0
