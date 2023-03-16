import re
import csv
import sys
import string
import unicodedata
import requests
import numpy as np
from unidecode import unidecode
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def preprocess_primary_name(name):
    name = re.sub(r'\s\(.*\)', '', name)
    return name

def check_latin_chars(s):
    for ch in s:
        if ch.isalpha():
            if 'LATIN' not in unicodedata.name(ch):
                return False
    return True

def format_matches(results):
    matches = []
    records = [result['organization'] for result in results['items']]
    for record in records:
        ror_id = record['id']
        primary_name = preprocess_primary_name(record['name'])
        aliases, labels, acronyms = record['aliases'], record['labels'], record['acronyms']
        if labels != []:
            labels = [label['label'] for label in labels if check_latin_chars(label['label'])]
        variants =  aliases + labels + acronyms
        address = record['addresses'][0]
        city_name = address['geonames_city']['city'] if 'city' in address['geonames_city'].keys() else None
        admin1_name = address['geonames_city']['geonames_admin1']['name'] if 'geonames_admin1' in address['geonames_city'].keys() else None
        country_name = record['country']['country_name']
        location = ' '.join([part for part in [city_name, admin1_name, country_name] if part != None])
        record_metadata = {'ror_id': ror_id, 'primary_name':primary_name, 'variant_names':variants, 'location':location}
        matches.append(record_metadata)
    return matches

def normalize_text(text, ws_flag=False):
    text = unidecode(text.lower())
    text = re.sub('-', ' ', text)
    if ws_flag == True:
        text = re.sub(r'[^\w\s]', '', text)
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    return text

def predict_ror_id_weighted(affiliation, matches):
    # Preprocess the affiliation matches
    text_data = [normalize_text(affiliation, True)] + [normalize_text(m["primary_name"] + " " + " ".join(m["variant_names"]) + " " + m["location"]) for m in matches]
    print(text_data)
    # Extract TF-IDF features from the affiliation matches
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_features = tfidf.fit_transform(text_data)

    # Split the features back into the affiliation and potential matches
    affiliation_features = tfidf_features[0]
    match_features = tfidf_features[1:]

    # Take into account the distribution of probable matches returned by the affiliation service
    # (80% likehood of a match in the first five results, with an incremental drop off subsequently), 
    # then apply the match weights based on the index of the match in the match set. 
    num_matches = len(matches)
    if num_matches <= 5:
        match_weights = [0.8 / (i + 1) for i in range(num_matches)]
    else:
        match_weights = [0.8 / (i + 1) for i in range(5)]
        for i in range(5, num_matches):
            weight = 0.8 / (5 + 2 * (i - 5))
            match_weights.append(weight)            
    match_weights = np.array(match_weights).reshape(-1, 1)

    # Compute the weighted match features
    match_features_weighted = match_features.multiply(match_weights)

    # Train a logistic regression model to classify the matches
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(match_features_weighted, [m["ror_id"] for m in matches])

    # Predict the ROR ID for the affiliation string
    predicted_ror_id = lr.predict(affiliation_features)[0]

    return predicted_ror_id

def get_best_match(affiliation_string, results):
    num_results = results['number_of_results']
    if num_results == 1:
        return results['items'][0]['organization']['id']
    elif num_results != 0:
        matches = format_matches(results)
        prediction = predict_ror_id_weighted(affiliation_string, matches)
        return prediction
    return 'No results'

def query_affiliation(affiliation, ror_id):
    try:
        affiliation_encoded = quote(affiliation)
        url = f"http://localhost:9292/organizations?affiliation={affiliation_encoded}"
        response = requests.get(url)
        results = response.json()
        best_match_prediction = get_best_match(affiliation, results)
        ror_id_in_results = False
        ror_id_chosen = False
        chosen_id = ''
        index = ''
        for i, item in enumerate(results["items"]):
            if item["organization"]["id"] == ror_id:
                ror_id_in_results = True
                index = str(i)
                if item["chosen"]:
                    ror_id_chosen = True
                    chosen_id = ror_id
                break
            elif item['chosen']:
                chosen_id = item["organization"]["id"]
        if ror_id_chosen == False:
            chosen_id = best_match_prediction
            if chosen_id == ror_id:
                ror_id_chosen = True
        return url, ror_id_in_results, ror_id_chosen, index, chosen_id
    except Exception:
        return '','','','','Error'


def parse_and_query(f):
    outfile = 'aps_test_results_first_five.csv'
    with open(outfile, 'w') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['aps_ror_id','doi','aps_institution','affiliation', 'query_url','in_results', 'chosen','index', 'chosen_id'])
    with open(f, 'r+', encoding='utf-8-sig') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            affiliation = row['affiliation']
            aps_ror_id = row['aps_ror_id']
            matches_aff_ror_id = False
            if aps_ror_id != '' and aps_ror_id != None:
                query_url, ror_id_in_results, ror_id_chosen, index, chosen_id = query_affiliation(affiliation, aps_ror_id)
                with open(outfile, 'a') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow(list(row.values()) + [query_url, ror_id_in_results, ror_id_chosen, index, chosen_id])


if __name__ == '__main__':
    parse_and_query(sys.argv[1])