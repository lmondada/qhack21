# qhack21
QHACK21: ESA collision warning predictions

I wrote this quickly to get everyone up to speed

## Load dataset
We are using the dataset described on the [ESA website](https://kelvins.esa.int/collision-avoidance-challenge/data/).
It might be useful to quickly go through their descriptions to familiarise yourself with the dataset.
Don't worry too much about each individual feature, as we will mostly focus on the covariance matrix for now.

Load dataset into Python:
```python
from utils.data_handling import load_data

# this will download the data from the remote server
# when run the first time
data = load_data()
```

For playing around, it suffices to consider a small subset of the training dataset, to speed things up.
```python
data = data[:100]
```

What we care about (for now) are the 3x3 combined covariance matrices that represent the observation uncertainty
in position. This is what we want to encode in our circuit.
We get it as
```python
from utils.data_handling import load_data, get_combined_cov_pos

data = load_data()[:100]
covs = get_combined_cov_pos(data)
```

## How to structure our work
TODO.
Luca will push some more files soon with some structure for our code that we can all work on.