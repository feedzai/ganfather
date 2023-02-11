import torch


class rs_CF(torch.nn.Module):
    def __init__(self, k=400, Nu=6040, version="mean_relu_diff"):
        super(rs_CF, self).__init__()
        self.k = k
        self.Nu = Nu
        self.version = version

    def forward(self, R):
        eps = 1e-5
        Rn = R / (((R ** 2).sum(1, True)) ** 0.5 + eps)
        D = torch.matmul(Rn, Rn.T)
        topk, indices = torch.topk(D, self.k)
        D = torch.zeros_like(D).scatter_(1, indices, topk)
        P = torch.matmul(D, R) / (torch.matmul(D, (R > 0).float()) + eps)
        P2 = torch.where(P > 0, P, R.sum(0) / ((R > 0).sum(0) + 1e-5))
        return P2

    def objective(self, R, mi):
        P = self.forward(R)
        if self.version == "mean_rating":
            goal = -P[: self.Nu, mi].mean()
        elif self.version == "top3000_mean_rating":
            topk = torch.topk(P[: self.Nu, mi], 3000)[1]
            goal = -P[topk, mi].mean()
        elif self.version == "top1000_mean_rating":
            topk = torch.topk(P[: self.Nu, mi], 1000)[1]
            goal = -P[topk, mi].mean()
        elif self.version == "top300_mean_rating":
            topk = torch.topk(P[: self.Nu, mi], 300)[1]
            goal = -P[topk, mi].mean()
        elif self.version == "mean_relu_diff":
            goal = (
                torch.relu(P[: self.Nu].clamp(min=1) - P[: self.Nu, [mi]]).sum(1).mean()
            )
        return goal

    def evaluate(self, Rt, Re):
        with torch.no_grad():
            P = self.forward(Rt)
            return (
                (
                    (P[: self.Nu].clamp(1, 5) - Re[: self.Nu]) ** 2
                    * (Re[: self.Nu] > 0)
                ).sum()
                / (Re[: self.Nu] > 0).sum()
            ) ** 0.5

    def mean_rating(self, R, mi):
        with torch.no_grad():
            P = self.forward(R)
            return P[: self.Nu, mi].mean(), P[self.Nu :, mi].mean(), P[:, mi].mean()

    def count_topK(self, R, mi, k):
        with torch.no_grad():
            P = self.forward(R)
            return ((P[: self.Nu] > P[: self.Nu, [mi]]).sum(1) < k).sum()
