import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mysql.connector import connect, Error
from textblob import Blobber
from textblob_fr import PatternTagger, PatternAnalyzer
import settings
import openai
import random
import requests
import string
import sys
import os
import re
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Importance, EmpiricalMarginal
from pyro.infer import MCMC, NUTS
from datetime import datetime, timedelta
import geoip2.database
from twilio.rest import Client
from dateutil.relativedelta import relativedelta
import torch.nn.functional as F

def load_data_from_mysql():
    settings.getSettings()
    thistime = "" + datetime.today().strftime('%d-%m-%Y %H:%M:%S')
    print(thistime + "   Acquisition des données: Erreurs du réseau")
    cnx = connect(user=settings.AlicIA_db_user, password=settings.AlicIA_db_pass,
                  host='alicemedia.ca', database='aliceid')
    cursor = cnx.cursor(buffered=True)
    query = "SELECT * FROM alicia.errors ORDER BY userid DESC"
    cursor.execute(query)
    results = cursor.fetchall()
    temp_error_logfile = "temp_error_logs.txt"
    with open(temp_error_logfile, "w", encoding="utf-8") as log_file:
        for row in results:
            try:
                err_userid = row[0]
                filename = row[1]
                erreur = row[2]
                date = row[3]
                user = row[4]
                line = int(row[5])
                start_line = max(0, line - 5)
                end_line = line + 5
                lines = []
                content = ''
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as file:
                        for i, line in enumerate(file, start=1):
                            if start_line <= i <= end_line:
                                lines.append(line.rstrip('\n'))
                            elif i > end_line:
                                break
                    content = '\n'.join(lines)
                    content = content.encode('utf-8').decode('utf-8')
                log_file.write(f"UserID: {err_userid}\n")
                log_file.write(f"Fichier: {filename}\n")
                log_file.write(f"Erreur: {erreur}\n")
                log_file.write(f"Date: {date}\n")
                log_file.write(f"Utilisateur: {user}\n")
                log_file.write(f"Ligne: {line}\n")
                log_file.write("Code snippet:\n")
                log_file.write(content)
                log_file.write("\n-----------------------\n")
            except Exception as e:
                print(f"{thistime}   Erreur lors du traitement des erreurs sur le réseau : {e}")
    cnx.close()
    print(thistime + "   Acquisition des données: Requêtes sur le réseau")
    cnx = connect(user=settings.AlicIA_db_user, password=settings.AlicIA_db_pass,
                  host='alicemedia.ca', database='aliceid')
    cursor = cnx.cursor(buffered=True)
    query = "SELECT * FROM alicia.logs ORDER BY userid DESC"
    cursor.execute(query)
    results = cursor.fetchall()
    temp_request_logfile = "temp_requests_logs.txt"
    with open(temp_request_logfile, "w", encoding="utf-8") as log_file:
        for row in results:
            try:
                req_userid = row[0]
                requete = row[2]
                date = row[3]
                user = row[1]
                log_file.write(f"UserID: {req_userid}\n")
                log_file.write(f"Requête: {requete}\n")
                log_file.write(f"Date: {date}\n")
                log_file.write(f"Utilisateur: {user}\n")
                log_file.write("\n-----------------------\n")
            except Exception as e:
                print(f"{thistime}   Erreur lors du traitement des requêtes sur le réseau : {e}")

    print(thistime + "   Acquisition des données: Activités sur le réseau")
    cnx = connect(user=settings.AlicIA_db_user, password=settings.AlicIA_db_pass,
                  host='alicemedia.ca', database='aliceid')
    cursor = cnx.cursor(buffered=True)
    query = "SELECT * FROM aliceid.activity ORDER BY userid DESC LIMIT 10000"
    cursor.execute(query)
    results = cursor.fetchall()
    temp_activity_logfile = "temp_activity_logs.txt"
    with open(temp_activity_logfile, "w", encoding="utf-8") as log_file:
        for row in results:
            try:
                act_userid = row[0]
                user = row[1]
                activity = row[2]
                date = row[3]
                ip = row[4]
                log_file.write(f"UserID: {act_userid}\n")
                log_file.write(f"Activité: {activity}\n")
                log_file.write(f"Date: {date}\n")
                log_file.write(f"Utilisateur: {user}\n")
                log_file.write(f"Adresse IP: {ip}\n")
                log_file.write("\n-----------------------\n")
            except Exception as e:
                print(f"{thistime}   Erreur lors du traitement des activités sur le réseau : {e}")

    print(thistime + "   Acquisition des données: Factures")
    cnx = connect(user=settings.AlicIA_db_user, password=settings.AlicIA_db_pass,
                  host='alicemedia.ca', database='aliceid')
    cursor = cnx.cursor(buffered=True)
    query = "SELECT * FROM facturation.factures ORDER BY userid DESC"
    cursor.execute(query)
    results = cursor.fetchall()
    temp_factures_logfile = "temp_factures_logs.txt"
    with open(temp_factures_logfile, "w", encoding="utf-8") as log_file:
        for row in results:
            try:
                fact_id = row[1]
                user = row[5]
                total = row[4]
                date = row[3]
                service = row[2]
                if service == 'AD':
                    service = 'Web Design'
                if service == 'AH':
                    service = 'Hébergement Web'
                if service == 'AM':
                    service = 'Marketing en ligne'
                log_file.write(f"ID: {fact_id}\n")
                log_file.write(f"Service: {service}\n")
                log_file.write(f"Date: {date}\n")
                log_file.write(f"Utilisateur: {user}\n")
                log_file.write(f"Total de la facture: {total}\n")
                log_file.write("\n-----------------------\n")
            except Exception as e:
                print(f"{thistime}   Erreur lors du traitement des factures : {e}")

    print(thistime + "   Acquisition des données: Dépenses")
    cnx = connect(user=settings.AlicIA_db_user, password=settings.AlicIA_db_pass,
                  host='alicemedia.ca', database='aliceid')
    cursor = cnx.cursor(buffered=True)
    query = "SELECT * FROM facturation.depenses ORDER BY userid DESC"
    cursor.execute(query)
    results = cursor.fetchall()
    temp_depenses_logfile = "temp_depenses_logs.txt"
    with open(temp_depenses_logfile, "w", encoding="utf-8") as log_file:
        for row in results:
            try:
                depense_id = row[1]
                category = row[2]
                total = row[6]
                depense = row[3]
                supplier = row[7]
                date = row[12]
                log_file.write(f"ID: {depense_id}\n")
                log_file.write(f"Date: {date}\n")
                log_file.write(f"Catégorie: {category}\n")
                log_file.write(f"Fournisseur: {supplier}\n")
                log_file.write(f"Dépenses: {depense}\n")
                log_file.write(f"Montant: {total}\n")
                log_file.write("\n-----------------------\n")
            except Exception as e:
                print(f"{thistime}   Erreur lors du traitement des dépenses : {e}")

    print(thistime + "   Acquisition des données: Utilisateurs du réseau")
    cnx = connect(user=settings.AlicIA_db_user, password=settings.AlicIA_db_pass,
                  host='alicemedia.ca', database='aliceid')
    cursor = cnx.cursor(buffered=True)
    query = "SELECT * FROM aliceid.users ORDER BY userid ASC"
    cursor.execute(query)
    results = cursor.fetchall()
    temp_users_logfile = "temp_users_logs.txt"
    with open(temp_users_logfile, "w", encoding="utf-8") as log_file:
        for row in results:
            try:
                userid = row[0]
                first_name = row[1]
                last_name = row[2]
                email = row[3]
                username = row[4]
                phone = row[9]
                address = row[10]
                ville = row[11]
                province = row[12]
                if province == 'Québec':
                    province = 'QC'
                if province == 'Quebec':
                    province = 'QC'
                if province == 'Ontario':
                    province = 'ON'
                if province == 'Georgia':
                    province = 'GA'
                if province == 'France':
                    province = 'FR'
                code_postal  = row[13]
                sec_lvl = row[14]
                log_file.write(f"UserID: {userid}\n")
                log_file.write(f"Username: {username}\n")
                log_file.write(f"Nom: {first_name} {last_name}\n")
                log_file.write(f"Courriel: {email}\n")
                log_file.write(f"Téléphone: {phone}\n")
                log_file.write(f"Adresse: {address} {ville} {code_postal}\n")
                log_file.write(f"Province: {province} \n")
                log_file.write(f"Accès: {sec_lvl}\n")
                log_file.write("\n-----------------------\n")
            except Exception as e:
                print(f"{thistime}   Erreur lors du traitement des utilisateurs du réseau : {e}")
    
    def parse_logfile_to_dataframe(logfile, columns):
        # Cette fonction peut être utilisée pour lire un fichier journal et le convertir en DataFrame
        data = []
        with open(logfile, 'r', encoding='utf-8') as file:
            entry = {}
            for line in file:
                if line.strip() == '-----------------------':
                    if entry:
                        data.append(entry)
                        entry = {}
                else:
                    try:
                        parts = line.strip().split(':', 1)
                        if len(parts) == 2:
                            key, value = parts
                            entry[key.strip()] = value.strip()
                        else:
                            msg = "Skipping line in {logfile}"        

                    except Exception as e:
                        print(f"{thistime}   Erreur lors du traitement de {value} : {e}")
                    
        return pd.DataFrame(data, columns=columns)
    
    error_columns = ["UserID", "Fichier", "Erreur", "Date", "Utilisateur", "Ligne"]
    error_logs = parse_logfile_to_dataframe(temp_error_logfile, error_columns)
    
    request_columns = ["UserID", "Requête", "Date", "Utilisateur"]
    request_logs = parse_logfile_to_dataframe(temp_request_logfile, request_columns)
    
    activity_columns = ["UserID", "Utilisateur", "Activité", "Date", "Adresse IP"]
    activity_logs = parse_logfile_to_dataframe(temp_activity_logfile, activity_columns)

    factures_columns = ["ID", "Service", "Date", "Utilisateur", "Total de la facture"]
    factures_logs = parse_logfile_to_dataframe(temp_factures_logfile, factures_columns)

    depenses_columns = ["ID", "Date", "Catégorie", "Fournisseur", "Dépenses", "Montant"]
    depenses_logs = parse_logfile_to_dataframe(temp_depenses_logfile, depenses_columns)

    users_columns = ["UserID", "Username", "Nom", "Courriel", "Téléphone", "Adresse", "Province", "Accès"]
    users_logs = parse_logfile_to_dataframe(temp_users_logfile, users_columns)

    os.remove(temp_error_logfile)
    os.remove(temp_users_logfile)
    os.remove(temp_request_logfile)
    os.remove(temp_activity_logfile)
    os.remove(temp_factures_logfile)
    os.remove(temp_depenses_logfile)
    
    return error_logs, request_logs, activity_logs, factures_logs, depenses_logs, users_logs

def requestsModel(users_logs, request_logs, error_logs, trend_analysis_act, trend_analysis_req, 
                  inference=True, load_model=False, save_model=False, 
                  model_path=None, encoders_path=None, first_run=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if first_run:
        # Entraînement des transformateurs pour la première fois
        tfidf_vectorizer = TfidfVectorizer()
        encoder = OneHotEncoder(handle_unknown='ignore')
        tfidf_vectors = tfidf_vectorizer.fit_transform(request_logs['Requête'])
        user_encoding = encoder.fit_transform(users_logs[['Username']])
    else:
        # Chargement des transformateurs sauvegardés
        if os.path.exists(encoders_path):
            tfidf_vectorizer, encoder = torch.load(encoders_path)
            tfidf_vectors = tfidf_vectorizer.transform(request_logs['Requête'])
            user_encoding = encoder.transform(users_logs[['Username']])
        else:
            raise FileNotFoundError("Le fichier des transformateurs sauvegardés n'a pas été trouvé.")

    # Suite du traitement des données
    user_encoding = np.zeros((tfidf_vectors.shape[0], user_encoding.shape[1]))  # À vérifier si nécessaire
    request_dates = pd.to_datetime(request_logs['Date'], format="%d-%m-%Y %H:%M:%S")
    hours = request_dates.dt.hour.values
    error_dates = pd.to_datetime(error_logs['Date'], format="%d-%m-%Y %H:%M:%S").dt.date
    daily_errors = request_dates.dt.date.apply(lambda date: sum(error_dates == date))
    error_users = set(error_logs['Utilisateur'])
    user_has_errors = request_logs['Utilisateur'].apply(lambda user: user in error_users).astype(int)
    average_trend_act = trend_analysis_act.mean()
    average_trend_req = trend_analysis_req.mean()
    input_features = np.hstack([
        tfidf_vectors.toarray(),
        user_encoding,
        hours[:, None],
        user_has_errors.values[:, None],
        np.full((tfidf_vectors.shape[0], 1), average_trend_act),
        np.full((tfidf_vectors.shape[0], 1), average_trend_req)
    ])
    feature_dim = input_features.shape[1]
    hidden_dim = 10
    #scaler = StandardScaler()
    #input_features_normalized = scaler.fit_transform(input_features)
    if first_run:
        # Définition et échantillonnage des paramètres directement avec leurs distributions
        w1 = pyro.sample("w1", dist.Normal(torch.zeros(hidden_dim, feature_dim, device=device), torch.ones(hidden_dim, feature_dim, device=device)))
        b1 = pyro.sample("b1", dist.Normal(torch.zeros(hidden_dim, device=device), torch.ones(hidden_dim, device=device)))
        w2 = pyro.sample("w2", dist.Normal(torch.zeros(1, hidden_dim, device=device), torch.ones(1, hidden_dim, device=device)))
        b2 = pyro.sample("b2", dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)))
    elif load_model and model_path:
        saved_model = torch.load(model_path)
        # Rééchantillonnage des paramètres basés sur les distributions enregistrées
        w1 = pyro.sample("w1", dist.Normal(saved_model['w1'], torch.ones_like(saved_model['w1']).to(device)))
        b1 = pyro.sample("b1", dist.Normal(saved_model['b1'], torch.ones_like(saved_model['b1']).to(device)))
        w2 = pyro.sample("w2", dist.Normal(saved_model['w2'], torch.ones_like(saved_model['w2']).to(device)))
        b2 = pyro.sample("b2", dist.Normal(saved_model['b2'], torch.ones_like(saved_model['b2']).to(device)))
    #regularization_strength = 0.01
    #regularization_term = regularization_strength * (torch.norm(w1) ** 2 + torch.norm(w2) ** 2)

    for i in range(len(request_logs)):
        max_val = 1000
        x = torch.tensor(input_features[i], dtype=torch.float32).to(device)     
        hidden_layer = F.relu(x @ w1.T + b1)
        #print("DEBUG MODE: Hidden layer values:", hidden_layer)
        clamped_layer = torch.clamp(hidden_layer @ w2.T + b2, max=max_val)
        expected_errors = torch.exp(clamped_layer)
        #print("DEBUG MODE: Expected errors:", expected_errors)
        if inference:
            obs_tensor = torch.tensor(daily_errors[i], dtype=torch.float32).to(device)
            pyro.sample(f'obs_{i}', dist.Poisson(expected_errors), obs=obs_tensor)
        else:
            pyro.sample(f'obs_{i}', dist.Poisson(expected_errors))

    # Sauvegarde des Poids et des Transformateurs
    if save_model and model_path:
        torch.save({'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}, model_path)
        torch.save((tfidf_vectorizer, encoder), encoders_path)
    return tfidf_vectorizer, encoder

def prepare_single_request(tfidf_vectorizer, encoder, error_users, request, trend_analysis_act, trend_analysis_req):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tfidf_vector = tfidf_vectorizer.transform([request['Requête']])
    this_user = request['Utilisateur']    
    num_user_columns = len(encoder.categories_[0])
    user_encoding = np.zeros((1, num_user_columns))
    user_index = np.where(encoder.categories_[0] == this_user)[0]
    if len(user_index) > 0:
        user_encoding[:, user_index] = 1  
    request_date = pd.to_datetime(request['Date'], format="%d-%m-%Y %H:%M:%S")
    hour = request_date.hour
    user_has_error = 1 if this_user in error_users else 0
    average_trend_act = trend_analysis_act.mean()
    average_trend_req = trend_analysis_req.mean()
    
    input_feature = np.hstack([
        tfidf_vector.toarray(),
        user_encoding,
        [[hour, user_has_error, average_trend_act, average_trend_req]]
    ])
    
    #scaler = StandardScaler()
    #input_feature_normalized = scaler.fit_transform(input_feature)
    input_feature_tensor = torch.tensor(input_feature, dtype=torch.float32).to(device)
    return input_feature_tensor


def predict_error_distribution(single_request_feature, posterior_samples, model_path, first_run):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    max_val = 1000
    if first_run:
        w1 = posterior_samples['w1'].to(device)
        b1 = posterior_samples['b1'].to(device)
        w2 = posterior_samples['w2'].to(device)
        b2 = posterior_samples['b2'].to(device)
    else:
        saved_model = torch.load(model_path)
        w1 = saved_model['w1'].to(device)
        b1 = saved_model['b1'].to(device)
        w2 = saved_model['w2'].to(device)
        b2 = saved_model['b2'].to(device)

    hidden_layer = F.relu(single_request_feature @ w1.T + b1)
    clamped_layer = torch.clamp(hidden_layer @ w2.T + b2, max=max_val)
    expected_errors = torch.exp(clamped_layer)
    
    return expected_errors.item()




def probabilistic_risk(scanned_user, scanned_request, users_logs, error_logs, request_logs, trend_analysis_act, trend_analysis_req, model_path, first_run, loaded_samples):
    try:
        error_users = set(error_logs['Utilisateur'])
        thistime = datetime.today().strftime('%d-%m-%Y %H:%M:%S')
        tfidf_vectorizer, encoder = requestsModel(users_logs, request_logs, error_logs, trend_analysis_act, trend_analysis_req, inference=False, load_model=not first_run, save_model=False, model_path='C:\\Users\\alicemedia\\parle\\test_weightsModel.pt', encoders_path='C:\\Users\\alicemedia\\parle\\test_encodersModel.pt', first_run=first_run)
        single_request = {
            'Date': thistime,
            'Utilisateur': scanned_user,
            'Requête': scanned_request
        }

        if encoder.categories_:
            num_user_columns = len(encoder.categories_[0])
            user_encoding = np.zeros((1, num_user_columns))
            user_index = np.where(encoder.categories_[0] == scanned_user)[0]
            if len(user_index) > 0:
                user_encoding[:, user_index] = 1
        else:
            user_encoding = np.zeros((1, 1))

        tfidf_vector = tfidf_vectorizer.transform([single_request['Requête']])
        single_request_feature = prepare_single_request(tfidf_vectorizer, encoder, error_users, single_request, trend_analysis_act, trend_analysis_req)
        mean_predicted_errors = predict_error_distribution(single_request_feature, loaded_samples, model_path=model_path, first_run=False)

        mean_predicted_errors = np.mean(mean_predicted_errors)
        percentile_5 = np.percentile(mean_predicted_errors, 5)
        percentile_95 = np.percentile(mean_predicted_errors, 95)

        print(f"{thistime}   Prédiction moyenne d'erreurs : {mean_predicted_errors}")
        print(f"{thistime}   Intervalle de confiance à 90% : ({percentile_5}, {percentile_95})")
        print(f"{thistime}   Détection d'une requête pouvant générer {mean_predicted_errors:.2f} erreur(s)")
        return mean_predicted_errors
    except Exception as e:
        print(f"{thistime}   Toutes mes excuses, je n'ai pas pu évaluer le risque de la requête. Erreur {str(e)}")
        error_speech = "Toutes mes excuses je n'ai pas pu évaluer le risque de la requête."
        cnx = connect(user=settings.AlicIA_db_user, password=settings.AlicIA_db_pass, host='alicemedia.ca', database='aliceid')
        cursor = cnx.cursor(buffered=True)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        logfile = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logfile = "C:\\Users\\alicemedia\\parle\\test_" + logfile
        logline = format(sys.exc_info()[-1].tb_lineno)
        logerror = error_speech + " Erreur: " + str(e)
        logtime = datetime.today().strftime('%d-%m-%Y %H:%M:%S')
        query = ("INSERT IGNORE INTO alicia.errors "
           "(username, error, date, file, line) "
           "VALUES ('canal4', %s, %s, %s, %s)")
        data_log = (logerror, logtime, logfile, logline)
        cursor.execute(query, data_log)
        cnx.commit()
        cursor.close()
        cnx.close()
        return False 

try:    
    settings.getSettings()
    thistime = datetime.today().strftime('%d-%m-%Y %H:%M:%S')
    error_logs, request_logs, activity_logs, factures_logs, depenses_logs, users_logs = load_data_from_mysql()
    print(thistime + "   Identification des règles d'association et des relations fréquentes")
    user_error_logs = pd.merge(error_logs, users_logs, left_on='Utilisateur', right_on='Username', suffixes=('_error', '_user'))
    user_request_logs = pd.merge(request_logs, users_logs, left_on='Utilisateur', right_on='Username', suffixes=('_request', '_user'))
    user_activity_logs = pd.merge(activity_logs, users_logs, left_on='Utilisateur', right_on='Username', suffixes=('_activity', '_user'))
    user_factures_logs = pd.merge(factures_logs, users_logs, left_on='Utilisateur', right_on='Username', suffixes=('_factures', '_user'))
    error_logs_copy = error_logs.copy()
    error_logs_copy['Date'] = pd.to_datetime(error_logs_copy['Date'], format="%d-%m-%Y %H:%M:%S")
    request_logs_copy = request_logs.copy()
    request_logs_copy['Date'] = pd.to_datetime(request_logs_copy['Date'], format="%d-%m-%Y %H:%M:%S")
    activity_logs_copy = activity_logs.copy()
    activity_logs_copy['Date'] = pd.to_datetime(activity_logs_copy['Date'], format="%d/%m/%Y %H:%M:%S")
    factures_logs_copy = factures_logs.copy()
    factures_logs_copy['Date'] = pd.to_datetime(factures_logs_copy['Date'], format="%d/%m/%Y %H:%M:%S")
    depenses_logs_copy = depenses_logs.copy()
    depenses_logs_copy['Date'] = pd.to_datetime(depenses_logs_copy['Date'], format='%d/%m/%Y')
    #error_logs_copy['Minute'] = error_logs_copy['Date'].dt.strftime("%d-%m-%Y %H:%M")
    #request_logs_copy['Minute'] = request_logs_copy['Date'].dt.strftime("%d-%m-%Y %H:%M")
    #activity_logs_copy['Minute'] = activity_logs_copy['Date'].dt.strftime("%d/%m/%Y %H:%M")
    #factures_logs_copy['Minute'] = factures_logs_copy['Date'].dt.strftime("%d/%m/%Y %H:%M")

    fact_merged_data = pd.merge(activity_logs_copy, factures_logs_copy, on='Date', suffixes=('_activity', '_factures'))
    req_merged_data = pd.merge(request_logs_copy, error_logs_copy, on='Date', suffixes=('_requests', '_error'))
    act_merged_data = pd.merge(activity_logs_copy, error_logs_copy, on='Date', suffixes=('_activity', '_error'))
    factures_onehot = pd.get_dummies(factures_logs, columns=["Service"])
    depenses_onehot = pd.get_dummies(depenses_logs, columns=["Catégorie"])
    users_onehot = pd.get_dummies(users_logs, columns=["Accès"])
    
    trend_analysis_fact = fact_merged_data.groupby('Date')['Total de la facture'].count()
    trend_analysis_req = req_merged_data.groupby('Date')['Requête'].count()
    trend_analysis_act = act_merged_data.groupby('Date')['Activité'].count()
    categories = ['Service_Hébergement Web', 'Service_Marketing en ligne', 'Service_Web Design']
    category_data = factures_onehot[categories].copy()
    category_data.loc[:, 'Total de la facture'] = factures_onehot['Total de la facture']
    category_data.loc[:, 'Total de la facture'] = pd.to_numeric(category_data['Total de la facture'], errors='coerce')
    category_impact_fact_mean = category_data.groupby(categories)['Total de la facture'].mean()
    category_impact_fact_sum = category_data.groupby(categories)['Total de la facture'].sum()
    categories = ['Catégorie_Deplacement', 'Catégorie_Equipement', 'Catégorie_Nourriture', 'Catégorie_Services']
    category_data = depenses_onehot[categories].copy()
    category_data.loc[:, 'Montant'] = depenses_onehot['Montant']
    category_data.loc[:, 'Montant'] = pd.to_numeric(category_data['Montant'], errors='coerce')
    category_impact_dep_mean = category_data.groupby(categories)['Montant'].mean()
    category_impact_dep_sum = category_data.groupby(categories)['Montant'].sum()
    access_categories = ['Accès_201105', 'Accès_Membre', 'Accès_admin', 'Accès_isens', 'Accès_limited', 'Accès_membre']
    access_category_data = users_onehot[access_categories].copy()
    access_category_data.loc[:, 'Activité'] = act_merged_data['Activité']
    access_category_data.loc[:, 'Activité'] = pd.to_numeric(access_category_data['Activité'], errors='coerce')
    access_activity_impact = access_category_data.groupby(access_categories)['Activité'].count()
    encoder = OneHotEncoder()
    user_encoding = encoder.fit_transform(users_logs[['Username']])
    average_trend_act = trend_analysis_act.mean()
    average_trend_req = trend_analysis_req.mean()
    user_encoding = torch.tensor(user_encoding, dtype=torch.float32).to(device)
    hours = torch.tensor(hours, dtype=torch.float32).to(device)
    user_has_errors = torch.tensor(user_has_errors.values, dtype=torch.float32).to(device)
    average_trend_act = torch.tensor(average_trend_act, dtype=torch.float32).to(device)
    average_trend_req = torch.tensor(average_trend_req, dtype=torch.float32).to(device)
    print(thistime + "   Modélisation de l'incertitude et des dépendances")
    first_run = settings.AlicIA_first_run
    nuts_kernel = NUTS(requestsModel)
    mcmc = MCMC(nuts_kernel, num_samples=1, warmup_steps=1)
    print(thistime + "   Création du Réseau Bayésien Probabiliste")
    mcmc.run(users_logs, request_logs, error_logs, trend_analysis_act, trend_analysis_req, inference=True, load_model=settings.AlicIA_load_model, save_model=True, model_path='C:\\Users\\alicemedia\\parle\\test_weightsModel.pt', encoders_path='C:\\Users\\alicemedia\\parle\\test_encodersModel.pt',  first_run=settings.AlicIA_first_run)
    samples = mcmc.get_samples()
    print(thistime + "   " + str(samples.keys()))
    torch.save(samples, "C:\\Users\\alicemedia\\parle\\test_requestsModel.pt")
    loaded_samples = torch.load("C:\\Users\\alicemedia\\parle\\test_requestsModel.pt")
	
    tfidf_vectorizer, encoder = requestsModel(users_logs, request_logs, error_logs, trend_analysis_act, trend_analysis_req, inference=False, load_model=settings.AlicIA_load_model, save_model=False, model_path='C:\\Users\\alicemedia\\parle\\test_weightsModel.pt', encoders_path='C:\\Users\\alicemedia\\parle\\test_encodersModel.pt', first_run=settings.AlicIA_first_run)
    utilisateur = 'samuel'
    element = 'Accès à la gestion des membres'
    model_path = 'C:\\Users\\alicemedia\\parle\\test_weightsModel.pt'
    mean_predicted_errors = probabilistic_risk(utilisateur, element, users_logs, error_logs_copy, request_logs_copy, trend_analysis_act, trend_analysis_req, model_path, first_run, loaded_samples)
except Exception as e:
    thistime = datetime.today().strftime('%d-%m-%Y %H:%M:%S')
    print(f"{thistime}   La création du réseau bayésien probabiliste a échouée. Erreur {str(e)}")






