import requests
import json
import unicodedata
import numpy as np
import nltk
import predict_flair
from flair.models import SequenceTagger
from ip2geotools.databases.noncommercial import DbIpCity
import re
import ipinfo
from datetime import datetime
from math import cos, asin, sqrt, pi
import stanza
# from dateutil.relativedelta import relativedelta


def get_bss(query):
    """
     Extract BSS code from query
    """
    # AAAABCDDDD/designation
    regex = "[0-9]{5}[a-zA-Z][0-9]{4}/[a-zA-Z0-9]+"
    match = re.findall(regex, query)
    return match
    if match:
        return re.group(0)
    return -1



def get_insee(location):
    """

    returns the insee codes of the location + type of location (commune, departement or region)
    if type region : return insee code of all departements of the region
    returns -1 if location does not exist

    """
    location_ = location.lower()
    print(location_)
    url = 'https://geo.api.gouv.fr/communes?nom={c}&fields=nom,code,codesPostaux,' \
          'codeDepartement,codeRegion,population&format=json&geometry=centre' \
        .format(c=location_)
    exists = len(json.loads(requests.get(url).text))
    similar_communes = []
    if exists > 0:
        codes = json.loads(requests.get(url).text)
        result = [code["code"] for code in codes \
                  if unicodedata.normalize('NFD', code["nom"].lower()).encode('ascii', 'ignore').decode(
                "utf-8") == unicodedata.normalize('NFD', location_).encode('ascii', 'ignore').decode("utf-8")]

        if len(result) > 0:
            return {"type": "commune", "code": result}
        else:
            similar_communes = [code["code"] for code in codes]

    url = 'https://geo.api.gouv.fr/departements?nom={c}&fields=nom,code' \
          '&format=json&geometry=centre' \
        .format(c=location_)
    exists = len(json.loads(requests.get(url).text))
    similar_departements = []
    if exists > 0:
        codes = json.loads(requests.get(url).text)
        result = [code["code"] for code in codes \
                  if unicodedata.normalize('NFD', code["nom"].lower()).encode('ascii', 'ignore').decode(
                "utf-8") == unicodedata.normalize('NFD', location_).encode('ascii', 'ignore').decode("utf-8")]
        return result
        if len(result) > 0:
            return {"type": "departement", "code": result}
        elif len(similar_communes) == 0:
            similar_departements = [code["code"] for code in codes]

    url = 'https://geo.api.gouv.fr/regions?nom={c}&fields=nom,code' \
          '&format=json&geometry=centre' \
        .format(c=location_)
    exists = len(json.loads(requests.get(url).text))
    similar_regions = []
    if exists > 0:
        codes = json.loads(requests.get(url).text)
        result = [code["code"] for code in codes \
                  if unicodedata.normalize('NFD', code["nom"].lower()).encode('ascii', 'ignore').decode(
                "utf-8") == unicodedata.normalize('NFD', location_).encode('ascii', 'ignore').decode("utf-8")]

        if len(result) > 0:
            url = "https://geo.api.gouv.fr/regions/{c}/departements".format(c=result[0])
            codes_ = json.loads(requests.get(url).text)
            result2 = [code["code"] for code in codes_]
            return {"type": "region", "code": result[0], "codes_departements": result2}

        elif len(similar_communes) == 0 and len(similar_departements) == 0:
            result3 = {}
            similar_regions = [code["code"] for code in codes]
            for similar_regions in similar_regions:
                url = "https://geo.api.gouv.fr/regions/{c}/departements".format(c=similar_region)
                codes_ = json.loads(requests.get(url).text)
                result3[similar_region] = [code["code"] for code in codes_]

    if len(similar_communes) > 0:
        return {"type": "commune", "code": similar_communes}
    elif len(similar_departements) > 0:
        return {"type": "departement", "code": similar_departements}
    elif len(similar_regions) > 0:
        return {"type": "region", "code": similar_regions, "codes_departements": result3}
    else:
        return -1


def insee_to_bss(code_location, type_location):
    """
    Returns the BSS codes corresponding to the INSEE code
    """
    if type_location == 'commune':
        url = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations?code_commune={c}&format=json&size=500".format(
            c=code_location)

    elif type_location == 'departement':
        url = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations?code_departement={c}&format=json&size=500".format(
            c=code_location)
    else:
        return -1

    exists = json.loads(requests.get(url).text)["count"]
    if exists > 0:
        data = json.loads(requests.get(url).text)
        bss = [station["code_bss"] for station in data["data"]]
        return bss

    else:
        return -1



def get_mesure_piezo(station, start_date=None, end_date=None):
    """
    Returns mesures corresponding to the station BSS code
    """
    url = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques?code_bss={bss}&date_debut_mesure={d1}&date_fin_mesure={d2}&size=1".format(
        bss=station, d1=start_date, d2=end_date)
    number = json.loads(requests.get(url).text)["count"]
    if number > 0:
        url = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques?code_bss={bss}&date_debut_mesure={d1}&date_fin_mesure={d2}&size={s}".format(
            bss=station, d1=start_date, d2=end_date, s=number + 1)
        return json.loads(requests.get(url).text)



# def POS_adj(text):
#     """
#     Returns words that are tagged ADJ in query
#     uses stanfordNLP POS server on port 9000
#     """
#     nlp = StanfordCoreNLP('http://localhost:9000')
#     splitted = text.split()
#     adjs = []
#
#     for word in splitted:
#         result = nlp.annotate(word,
#                               properties={# def POS_adj(text):
#     """
#     Returns words that are tagged ADJ in query
#     uses stanfordNLP POS server on port 9000
#     """
#     nlp = StanfordCoreNLP('http://localhost:9000')
#     splitted = text.split()
#     adjs = []
#
#     for word in splitted:
#         result = nlp.annotate(word,
#                               properties={
#                                   'annotators': 'pos',
#                                   'outputFormat': 'json',
#                                   'timeout': 1000,
#                               })
#         if result["sentences"][0]["tokens"][0]["pos"] == "ADJ":
#             adjs.append(result["sentences"][0]["tokens"][0]["word"])
#
#     return adjs

#                                   'annotators': 'pos',
#                                   'outputFormat': 'json',
#                                   'timeout': 1000,
#                               })
#         if result["sentences"][0]["tokens"][0]["pos"] == "ADJ":
#             adjs.append(result["sentences"][0]["tokens"][0]["word"])
#
#     return adjs


def POS_adj2(text):
    """
    Returns words that are tagged ADJ in query
    uses stanfordNLP POS with the python wrapper stanza
    """
    #     stanza.download('fr')
    nlp = stanza.Pipeline('fr', processors='tokenize, pos')
    splitted = text.split()
    adjs = []
    result = nlp(text)
    for sentence in result.sentences:
        for word in sentence.words:
            if word.pos == "ADJ":
                adjs.append(word.text)
    return adjs


def stem(word):
    """ stemming """
    word_ = "".join(list(word)[-4:])
    return word[:-4] + re.sub(r'iens|ains|ards|ain|ien|ard|ois|oi|ens|en|ais|ai|ins|in|s$', '', word_, count=1)


def get_location_from_adj(c, communes):
    """
    Returns the most similar commune to the adjective c from the list of communes
    """
    c_ = stem(c)
    dist = []
    for a in communes:
        limit = min(int(2 * len(c) / 3), len(a))
        a1 = "".join(list(a)[:limit])
        c1 = "".join(list(c)[:limit])
        d1 = nltk.edit_distance(c1, a1)
        d2 = nltk.edit_distance(c_, a)
        dist.append([d1, d2])

    dist = np.array(dist)
    avg = 0.5 * dist[:, 0] + 0.5 * dist[:, 1]
    sorted_ = np.argsort(avg)

    commune_ = communes[sorted_[0]]
    return commune_



def get_geolocation(ip_address):
    """
    Get location from ip adress
    """
    response = DbIpCity.get(ip_address, api_key='free')
    return response.city


def get_geolocation_ipinfo(ip_address):
    """
    Get location from ip adress with ipinfo API
    """
    access_token = 'ea47e58acb96e4'
    handler = ipinfo.getHandler(access_token)
    details = handler.getDetails(ip_address)
    city = details.city
    return city


def get_locations(query, communes, ip_address=None):
    """
    Use NER to extract locations from query,
    if NER gives no result, look for demonyms and return corresponding location,
    if none found, return geolocation
    """

    batch_size = 4
    MODEL_PATH = "NER_tool/stacked-standard-flair-150-wikiner.pt"
    tag_type = "label"
    model = SequenceTagger.load(MODEL_PATH)

    snippets = [[1, query]]
    result = predict_flair.get_entities(snippets, model, tag_type, batch_size)["snippets"][0][1]
    locations = [entity["text"] for entity in result["entities"] if "LOC" in str(entity["labels"][0])]
    if len(locations) > 0:
        return locations

    else:
        adjs = POS_adj2(query)
        locs = [get_location_from_adj(adj, communes) for adj in adjs]  # get_location_from_adj will be modified
        # to work with the dictionnary, for now it
        # uses string similarity
        if len(locs) > 0:
            return locs
        else:
            return get_geolocation(ip_address)



def distance(lon1, lat1, lon2, lat2):
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))  # 2*R*asin...



def get_coordinates(bss):
    url = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations?code_bss={c}&format=json&size=50".format(c=bss)
    exists = json.loads(requests.get(url).text)["count"]
    if exists > 0:
        data = json.loads(requests.get(url).text)
        info = data["data"][0]
        return {"code_commune": info["code_commune_insee"], "code_departement": info["code_departement"],
                "long": info["geometry"]["coordinates"][0], "lat": info["geometry"]["coordinates"][1],
                "date_fin_mesure": info["date_fin_mesure"]}
    else:
        return -1  # bss code does not correspond to any station


def get_closest_stations(bss, N=4):
    info = get_coordinates(bss)
    if info != -1:
        dep, long, lat = info["code_departement"], info["long"], info["lat"]
        dep_stations = insee_to_bss(dep, "departement")
        dist = {}

        for station in dep_stations:
            _ = get_coordinates(station)
            long_, lat_, date = _["long"], _["lat"], _["date_fin_mesure"]
            if date is not None:
                date = date.split("-")
                last_mesure_date = datetime(int(date[0]), int(date[1]), int(date[2]))

                if last_mesure_date >= datetime(2005, 1, 1):  # the last mesure date must be later than 01-01-2005
                    dist[station] = distance(long, lat, long_, lat_)

        sortd = dict(sorted(dist.items(), key=lambda item: item[1]))
        return list(sortd.keys())[: min(N, len(sortd.keys()))]

    return -1

locations = get_locations("A quelle profondeur se trouve la nappe à l'adresse 12 rue de Coulmiers, 45000 Orléans")
print(locations)