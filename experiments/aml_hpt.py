import pickle

import numpy as np
import optuna
import torch

import src.GAN as GAN
import src.utils as AML_objective
from src.aml_rules_proxy import Proxy


def objective(trial):

    # Hyperparameters
    alpha = 1
    beta = trial.suggest_float("beta", 100, 10000, log=True)
    gamma = trial.suggest_float("gamma", 100, 10000, log=True)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    t_sche = [1, 5]
    batch_size = 32
    optim_beta1 = 0.66
    gamma_scheduler = 1
    gan_type = "WGAN"
    clamp_lim = 0.01
    objective_version = "geoMean"
    proxy_type = "min"
    path_real = "PATH/TO/DATA"
    save_state = "Last"
    n_epochs = 100
    batch_per_epoch = 256
    batch_per_print = 16
    (M, S, D, T) = [5, 5, 5, 320]

    # Saving hyperparameters from each trial in a yaml file
    F = open(f"yamls/aml_{trial.number}.yaml", "a+")
    F.write(f"name: aml_{trial.number}\n")
    hp_names = [
        "alpha",
        "beta",
        "gamma",
        "lr",
        "t_sche",
        "batch_size",
        "optim_beta1",
        "gamma_scheduler",
        "gan_type",
        "clamp_lim",
        "objective_version",
        "proxy_type",
        "path_real",
        "save_state",
        "n_epochs",
    ]
    hp_values = [
        alpha,
        beta,
        gamma,
        lr,
        t_sche,
        batch_size,
        optim_beta1,
        gamma_scheduler,
        gan_type,
        clamp_lim,
        objective_version,
        proxy_type,
        path_real,
        save_state,
        n_epochs,
    ]
    for i in range(len(hp_names)):
        F.write(f"{hp_names[i]}: {hp_values[i]}\n")
    F.close()

    # Auxiliary variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ones = torch.ones(batch_size).to(device)
    zeros = torch.zeros(batch_size).to(device)
    one = torch.Tensor([1]).to(device)
    mone = one * -1
    BCE = torch.nn.BCELoss()
    best_optuna_epoch = 0
    best_optuna_objective = 0

    # Fetch real data
    train, validation = pickle.load(open(path_real, "rb"))
    train = torch.tensor(train).to(device)
    validation = torch.tensor(validation).to(device)

    # Initialize GAN objects
    g = GAN.G(batch_size).to(device)
    d = GAN.D(batch_size, gan_type == "WGAN").to(device)
    optimizerG = torch.optim.Adam(g.parameters(), lr=lr, betas=(optim_beta1, 0.999))
    optimizerD = torch.optim.Adam(d.parameters(), lr=lr, betas=(optim_beta1, 0.999))
    p = Proxy((batch_size * M, S, D, T), proxy_type).to(device)

    # Training loop
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}:", end=" ")
        for batch in range(batch_per_epoch):

            # Train Discriminator
            for par in g.parameters():
                par.requires_grad = False
            for par in d.parameters():
                par.requires_grad = True
            for d_iter in range(t_sche[1]):
                optimizerD.zero_grad()

                # Train with real
                real = train[np.random.randint(len(train), size=batch_size)]
                d_real = d(real, (real > 0).float())

                # Train with fake
                noise = torch.rand((batch_size, 100)).to(device)
                fake_a, fake_p = g(noise)
                fake, fake_c = g.sample(fake_a, fake_p)
                d_fake = d(fake, fake_c)

                if gan_type == "GAN":
                    d_loss_real = BCE(d_real, zeros)
                    d_loss_real.backward()
                    d_loss_fake = BCE(d_fake, ones)
                    d_loss_fake.backward()
                if gan_type == "WGAN":
                    d_loss_real = d_real.mean().view(1)
                    d_loss_real.backward(one)
                    d_loss_fake = d_fake.mean().view(1)
                    d_loss_fake.backward(mone)

                optimizerD.step()

                if gan_type == "WGAN":
                    for par in d.parameters():
                        par.data.clamp_(-clamp_lim, clamp_lim)

            # Train Generator
            for par in g.parameters():
                par.requires_grad = True
            for par in d.parameters():
                par.requires_grad = False
            for g_iter in range(t_sche[0]):
                optimizerG.zero_grad()

                # Generate data
                noise = torch.rand((batch_size, 100)).to(device)
                fake_a, fake_p = g(noise)
                fake, fake_c = g.sample(fake_a, fake_p)

                # Optimization objective loss
                optim_loss = AML_objective(fake, beta, objective_version)

                # Discriminator loss
                d_fake = d(fake, fake_c)
                if gan_type == "GAN":
                    disc_loss = BCE(d_fake, zeros)
                if gan_type == "WGAN":
                    disc_loss = d_fake.sum()

                # Alert loss
                p_fake = p(
                    fake.view(batch_size * M, S + D, T),
                    fake_c.view(batch_size * M, S + D, T),
                )
                proxy_loss = p_fake.sum()

                # Aggregated generator loss
                loss = gamma * proxy_loss + beta * disc_loss - alpha * optim_loss
                loss.backward()
                optimizerG.step()

        # Gather statistics on amount of money flowing and number of alerts triggered
        aflow_l = []
        triggers_l = []
        with torch.no_grad():
            for batch in range(batch_per_print):
                noise = torch.rand((batch_size, 100)).to(device)
                fake_a, fake_p = g(noise)
                fake, fake_c = g.sample(fake_a, fake_p)

                p_fake = p(
                    fake.view(batch_size * M, S + D, T),
                    fake_c.view(batch_size * M, S + D, T),
                )
                p_fake = p_fake.view(batch_size, M, 5, T).detach().cpu().numpy()

                for i in range(batch_size):
                    faki = fake[i].detach().cpu().numpy()
                    aflow_l += [
                        sum([int(min(np.sum(j[:S]), np.sum(j[-D:]))) for j in faki])
                    ]
                    triggers_l += [(p_fake[i] > 0).sum((0, 2))]

        # Save model snapshots and perform optuna related operations
        optuna_objective = np.mean(aflow_l) / max(1.0, np.mean(np.sum(triggers_l, 1)))
        print(optuna_objective)

        if optuna_objective > best_optuna_objective:
            best_optuna_epoch = epoch
            best_optuna_objective = optuna_objective
            torch.save(g.state_dict(), f"saves/aml_{trial.number}_G.pt")
            torch.save(d.state_dict(), f"saves/aml_{trial.number}_D.pt")

        if save_state != "Last" and (epoch + 1) % int(save_state) == 0:
            torch.save(g.state_dict(), f"saves/aml_{trial.number}_{epoch + 1}_G.pt")
            torch.save(d.state_dict(), f"saves/aml_{trial.number}_{epoch + 1}_D.pt")

        trial.report(optuna_objective, epoch)
        if trial.should_prune() or (epoch >= 20 and epoch >= best_optuna_epoch + 10):
            raise optuna.exceptions.TrialPruned()

    return best_optuna_objective


def main():
    storage = optuna.storages.RDBStorage(url="sqlite:///aml.db")
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    sampler = optuna.samplers.RandomSampler()
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=20, n_warmup_steps=15, n_min_trials=5
    )
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        study_name="aml",
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100)

    return 0


if __name__ == "__main__":
    main()
