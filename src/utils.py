import torch


def AML_objective(fake, beta=0.9, version="geoMean"):
    """
    Optimization objective to be use when training G in AML setting

    Parameters
    ----------
    fake: torch.tensor((batch,M,S+D,T)) or torch.tensor((M,S+D,T))
        Generated tensor example
    beta: float
        Coefficient of time consistency term
    version: string
        Which version of the objective we want:
         - 'geoMean' (default)
         - 'thesis'
         - 'cumsum'

    Returns
    -------
    float
        Value of AML objective
    """

    if len(fake.shape) == 3:
        fake.unsqueeze_(0)

    if version == "thesis":
        flow_in = fake[:, :, :5].sum((2, 3))
        flow_out = fake[:, :, 5:].sum((2, 3))
        loss = (flow_in + flow_out) - beta * torch.abs(flow_in - flow_out)
    elif version == "cumsum":
        eps = 1e-6
        money_flowing = torch.min(
            fake[:, :, :5].sum((2, 3)), fake[:, :, 5:].sum((2, 3))
        )
        mean_amount = (fake.sum((2, 3)) / ((fake > 0).sum((2, 3)) + eps)).detach() + eps
        balance_dif = fake[:, :, :5].sum(2) - fake[:, :, 5:].sum(2)
        balance_norm = balance_dif.cumsum(-1) / mean_amount.unsqueeze(-1)
        loss = money_flowing - beta * (balance_norm ** 2).mean(-1)
    else:
        flow_in = fake[:, :, :5].sum((2, 3))
        flow_out = fake[:, :, 5:].sum((2, 3))
        loss = torch.sqrt((flow_in + 1) * (flow_out + 1))

    return loss.mean()
