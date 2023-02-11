The GANfather: Controllable generation of malicious activity to improve defence systems
===============


Code accompanying the paper ["The GANfather: Controllable generation of malicious activity to improve defence systems"](https://openreview.net/group?id=KDD.org/2023/Conference/Applied_Data_Science_Track) (ADD LINK WHEN AVAILABLE).


## Code structure

The ``src`` folder contains:
- the Generator and Discriminator architectures for both use cases;
- the skeleton of the rules proxy network from the anti-money laundering use case, but the forward method was removed because the rules' logic is confidential;
- the recommender system.

The ``experiments`` folder contains the hyperparameter tuning executables from each use case.

For information regarding the versions of the packages we used, please refer to the `environment.yml` file.


## Data

The data used for the anti-money laundering use case is confidentual and as such cannot be published.

The data used for the recommender system was the MovieLens-1M dataset that can be found [here](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset).
