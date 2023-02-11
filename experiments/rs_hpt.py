import pickle

import numpy as np
import optuna
import torch

import src.rs_GAN as GAN
from src.rs_RS import rs_CF


def objective(trial):

    # Hyperparameters
    alpha = trial.suggest_float("alpha", 1e-5, 1, log=True)
    beta = 1
    objective_version = "mean_relu_diff"
    width = 128
    half_depth = 2
    lr_gan = 3e-5
    t_sche = [1, 2]
    layer_type = GAN.ResidualBlock
    batch_size = 1024
    gan_type = "WGAN"
    clamp_lim = 0.01
    path_real = "PATH/TO/DATA"
    save_state = 20
    n_epochs = 120
    batch_per_epoch = 1024
    batch_per_print = 32
    (Nu, Nm, Nf, K) = [6040, 3706, trial.suggest_categorical("Nf", [30, 60, 120]), 16]
    mi = 2390

    # Saving hyperparameters from each trial in a yaml file
    F = open(f"yamls/rs_{trial.number}.yaml", "a+")
    F.write(f"name: rs_{trial.number}\n")
    hp_names = [
        "alpha",
        "beta",
        "width",
        "half_depth",
        "lr_gan",
        "t_sche",
        "layer_type",
        "batch_size",
        "objective_version",
        "n_epochs",
        "Nf",
    ]
    hp_values = [
        alpha,
        beta,
        width,
        half_depth,
        lr_gan,
        t_sche,
        layer_type,
        batch_size,
        objective_version,
        n_epochs,
        Nf,
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
    best_optuna_objective = -1000

    # Fetch real data
    train, Re = pickle.load(open(path_real, "rb"))
    train = torch.tensor(train).to(device)
    Re = torch.tensor(Re).to(device)

    # Initialize GAN objects
    g = GAN.G([128] + [width] * half_depth, [width] * half_depth + [64], layer_type).to(
        device
    )
    d = GAN.D(
        [64] + [width] * (half_depth - 1) + [width // 2],
        [width] * (half_depth + 1),
        gan_type == "WGAN",
        layer_type,
    ).to(device)
    optimizerG = torch.optim.Adam(g.parameters(), lr=lr_gan)
    optimizerD = torch.optim.Adam(d.parameters(), lr=lr_gan)
    rs = rs_CF(Nu).to(device)

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
                noise = torch.randn((batch_size, 128)).to(device)
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
                noise = torch.randn((Nf, 128)).to(device)
                fake_a, fake_p = g(noise)
                fake, fake_c = g.sample(fake_a, fake_p)

                # Optimization objective loss
                R = torch.cat((train, fake))
                optim_loss = rs.objective(R, mi)
                optim_loss.backward()

                # Generate data
                noise = torch.randn((batch_size, 128)).to(device)
                fake_a, fake_p = g(noise)
                fake, fake_c = g.sample(fake_a, fake_p)

                # Discriminator loss
                d_fake = d(fake, fake_c)
                if gan_type == "GAN":
                    disc_loss = BCE(d_fake, zeros)
                if gan_type == "WGAN":
                    disc_loss = d_fake.mean()
                disc_loss.backward()

                # Aggregated generator loss
                loss = alpha * optim_loss + beta * disc_loss
                loss.backward()
                optimizerG.step()

        # Gather statistics on how much the movie is being recommended
        top10_l = []
        top50_l = []
        top300_l = []
        with torch.no_grad():
            for batch in range(batch_per_print):
                noise = torch.randn((Nf, 128)).to(device)
                fake_a, fake_p = g(noise)
                fake, fake_c = g.sample(fake_a, fake_p)
                R = torch.cat((train, fake))

                top10_l += [rs.count_topK(R, mi, 10).item()]
                top50_l += [rs.count_topK(R, mi, 50).item()]
                top300_l += [rs.count_topK(R, mi, 300).item()]

        # Save model snapshots and perform optuna related operations
        optuna_objective = (
            np.mean(top10_l) * 10 + np.mean(top50_l) * 3 + np.mean(top300_l)
        )
        print(optuna_objective)

        if optuna_objective > best_optuna_objective:
            best_optuna_epoch = epoch
            best_optuna_objective = optuna_objective
            torch.save(g.state_dict(), f"saves/rs_{trial.number}_G.pt")
            torch.save(d.state_dict(), f"saves/rs_{trial.number}_D.pt")

        if save_state == "Last":
            torch.save(g.state_dict(), f"saves/rs_{trial.number}_last_G.pt")
            torch.save(d.state_dict(), f"saves/rs_{trial.number}_last_D.pt")
        elif (epoch + 1) % int(save_state) == 0:
            torch.save(g.state_dict(), f"saves/rs_{trial.number}_{epoch + 1}_G.pt")
            torch.save(d.state_dict(), f"saves/rs_{trial.number}_{epoch + 1}_D.pt")

        trial.report(optuna_objective, epoch)

    return best_optuna_objective


def main():
    storage = optuna.storages.RDBStorage(url="sqlite:///rs.db")
    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        study_name="rs",
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100)

    return 0


if __name__ == "__main__":
    main()
