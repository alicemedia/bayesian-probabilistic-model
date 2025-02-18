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
pip install pandas
pip install scikit-learn
pip install mlxtend
pip install mysql-connector-python
pip install textblob
pip install textblob-fr
pip install openai
pip install requests
pip install numpy
pip install beautifulsoup4
pip install torch
pip install pyro-ppl
pip install geoip2
pip install twilio
pip install python-dateutil


# Use
Run "python learn_requestsModel.py" to generate encodersModel.pt, weightsModel.pt and requestsModel.pt in the working directory. Those library can be used with Python Torch code like this:
    single_request = {
        'Date': thistime,
        'Utilisateur': scanned_user,
         'Requête': scanned_request
    }
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    merged_data = pd.merge(request_logs, users_logs, left_on='Utilisateur', right_on='Username', how='left')
    tfidf_vectorizer = TfidfVectorizer()
    encoder = OneHotEncoder(handle_unknown='ignore')
    tfidf_vectors = tfidf_vectorizer.fit_transform(merged_data['Requête'])
    user_encoding = encoder.fit_transform(merged_data[['Username']])
        w1 = pyro.sample("w1", dist.Normal(torch.zeros(hidden_dim, feature_dim, device=device), torch.ones(hidden_dim, feature_dim, device=device)))
        b1 = pyro.sample("b1", dist.Normal(torch.zeros(hidden_dim, device=device), torch.ones(hidden_dim, device=device)))
        w2 = pyro.sample("w2", dist.Normal(torch.zeros(1, hidden_dim, device=device), torch.ones(1, hidden_dim, device=device)))
        b2 = pyro.sample("b2", dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)))
    daily_errors_tensor = torch.tensor(daily_errors, dtype=torch.float32).to(device)
    for i in range(len(request_logs)):
        max_val = 1000
        x = input_features[i].clone().detach()
        hidden_layer = F.relu(x @ w1.T + b1)
        clamped_layer = torch.clamp(hidden_layer @ w2.T + b2, max=max_val)
        expected_errors = torch.exp(clamped_layer)
        if inference:
            obs_tensor = daily_errors_tensor[i].clone().detach()
            pyro.sample(f'obs_{i}', dist.Poisson(expected_errors), obs=obs_tensor)
        else:
            pyro.sample(f'obs_{i}', dist.Poisson(expected_errors))

