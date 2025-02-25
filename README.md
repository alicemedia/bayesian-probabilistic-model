# bayesian-probabilistic-model
A Bayesian probabilistic AI model designed to estimate the likelihood of a specific request resulting in an error, by analyzing historical network activity and request patterns.

by Alice Media 2024-2025


# Install
First, clone the repository to your local machine: "git clone https://github.com/alicemedia/bayesian-probabilistic-model.git"

Install the required Python packages using pip: "pip install -r requirements.txt"

Alternatively, you can install the required packages manually: pip install pandas scikit-learn mlxtend mysql-connector-python textblob textblob-fr openai requests numpy beautifulsoup4 torch pyro-ppl geoip2 twilio python-dateutil


## Setup
Ensure you have the correct settings in place before using the model. Edit the settings.py file to configure necessary API keys, database credentials, or any custom configurations required for the project


# Dependancies
Requires the following Python libraries: pandas, scikit-learn, mlxtend, mysql-connector-python, textblob, textblob-fr, openai, requests, numpy, beautifulsoup4, torch, pyro-ppl, geoip2, twilio, python-dateutil.

Those Python libraries can be installed using pip by running: "pip install -r requirements.txt"


# Use
Run "python learn_requestsModel.py" to generate encodersModel.pt, weightsModel.pt and requestsModel.pt in the working directory. Those library can then be used with Python Torch.
