"""
A quick script to pull some data from flickr surrounding given points.
"""
import json
import requests
import time
import datetime
import fiona
from shapely.geometry import Point, shape
import pickle
import psycopg2, psycopg2.extras
import random

## scikitlearn stuff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import math

API_KEY = open("api_key.txt", 'r').read().strip()
NY_LAT = 40.69
NY_LON = -74.00
DC_LAT = 38.90
DC_LON = -77.00

BRONX_LON = -73.86
BRONX_LAT = 40.85
STISL_LON = -74.14
STISL_LAT = 40.58


def get_payload(city_lat, city_lon, c_page, min_date):
    """
    Get a number of photos within 32km of a point.
    :param city_lat: latitude in DD
    :param city_lon: longitude in DD
    :param c_page:  which page do you want? 1-n
    :return:
    """
    start = time.time()
    threshold = 1.1  # number of seconds we want a call to take
    payload = {"method": "flickr.photos.search",
               "accuracy": 8,
               "lat": city_lat,
               "lon": city_lon,
               "radius": 32,
               "extras": "date_upload,date_taken,owner_name,geo,tags",
               "per_page": 250,
               "min_upload_date": min_date,
               "max_upload_date": min_date+86400,
               "format": "json",
               "nojsoncallback": 1,
               "page": c_page,
               "api_key": API_KEY}
    flickr_req = requests.get("https://api.flickr.com/services/rest/",
                              params=payload)
    photo_list = json.loads(flickr_req.text)
    tot_time = time.time() - start
    if tot_time < threshold:
        time.sleep(threshold - tot_time)
    return photo_list["photos"]["photo"]


def multi_day_authors(in_file, num_days):
    """
    find all the authors in in_file that have been in the city for at least
    num_days
    :param in_file: string
    :param num_days: int
    :return:
    """
    valid_authors = set()
    checking_authors = dict()
    flickr_file = open(in_file, 'r')
    for line in flickr_file:
        in_json = json.loads(line)
        photo_time = time.strptime(
            in_json['datetaken'],
            '%Y-%m-%d %H:%M:%S')
        photo_time = time.mktime(photo_time)
        owner = in_json['owner']
        if owner not in valid_authors:
            if owner not in checking_authors:
                checking_authors[owner] = dict()
                checking_authors[owner]['oldest'] = photo_time
                checking_authors[owner]['newest'] = photo_time
            else:
                if photo_time > checking_authors[owner]['oldest']:
                    checking_authors[owner]['oldest'] = photo_time
                if photo_time < checking_authors[owner]['newest']:
                    checking_authors[owner]['newest'] = photo_time
                newest = checking_authors[owner]['newest']
                oldest = checking_authors[owner]['oldest']
                if oldest-newest > 0:
                    new_ts = datetime.datetime.fromtimestamp(newest)
                    old_ts = datetime.datetime.fromtimestamp(oldest)
                    delta = old_ts - new_ts
                    if delta.days >= num_days:
                        valid_authors.add(owner)

    # print len(valid_authors), len(checking_authors)
    flickr_file.close()
    return valid_authors


def get_borough_item(json_line, borough_geoms, stopwords):
    """
    given a line of JSON data, return the borough and JSON line, if applicable.
    If no borough, return a -1 code .
    :param json_line: JSON flickr object
    :param borough_geoms: Geometry objects for boroughs
    :return: (borough_key, JSON flickr object)
    """
    geoms = borough_geoms

    lat = json_line['latitude']
    lon = json_line['longitude']
    # It pains me to do the below two lines, but sometimes flickr
    # sends back geos that match NYC but the original data is bad
    # I think it has to do with their indexing being diffrent from
    # actual values?  But anyways, we need the data and the tags
    # make sense for NYC.
    if lon > 0:
        lon *= -1
    tgt_point = Point(lon, lat)  # silly shapely - Point(x,y,z)

    ret_val = list()

    # if not tgt_point.intersects(not_nyc_shp):
    for index, geo in enumerate(geoms):
        if tgt_point.intersects(geo[0]):
            ret_val.append(index)
            break

    if len(ret_val) > 0:
        tags = json_line['tags'].lower()
        # Tokenize, deduplicate, strip, restring - all one step.
        tags = ' '.join(set([_ for _ in tags.split(" ") if len(_.strip()) > 0]))
        for sw in stopwords:
            tags = tags.replace(sw, "")
        if len(tags) > 20:  # more than 20 chars
            ret_val.append(tags)
            return ret_val
    else:
        return False


def collect_nyc_training_data(thr_vals):
    """
    collect a minimum of x thresholds for each borough
    :param thr_vals:
    :return:
    """
    boroughs = fiona.open('./shp/nybb_wgs84.shp', 'r')
    geoms = list()
    for item in boroughs:
        shply_shape = shape(item['geometry'])
        adder = [shply_shape, item['properties']['BoroName']]
        geoms.append(adder)
        print adder

    if len(thr_vals) != len(geoms):
        return -1

    swfile = file("./stopwords.txt", 'r')
    stopwords = list()
    for ln in swfile:
        stopwords.append(ln.lower().strip())
    swfile.close()

    retvals = [list() for _ in range(0, len(thr_vals))]
    curr_id = 0

    curs = CONNECTION.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # We create a loop that watches to see if the lengths of the ret_vlas
    # is the same as the threshold item.
    # We get things from the database and see where they are from.

    done = False

    while done is False:
        dq = [len(_) for _ in retvals]
        print time.asctime(), dq, curr_id
        done = True
        for index, k in enumerate(thr_vals):
            if k != len(retvals[index]):
                done = False
        sql = "SELECT id, internal_id, tags, latitude, longitude " \
              "FROM flickr_data WHERE internal_id > %s " \
              "ORDER BY internal_id ASC LIMIT 10000"

        data = (curr_id, )
        curs.execute(sql, data)
        res = curs.fetchall()
        if len(res) == 0:
            print "Not enough data."
            return retvals
        for r in res:
            curr_id = r['internal_id']
            if r['latitude'] > 40:
            # if True:
                check_data = get_borough_item(r, geoms, stopwords)
            else:
                check_data = None
            if isinstance(check_data, list):
                cd = check_data[0]
                if 0 <= cd < len(retvals):
                    if len(retvals[cd]) < thr_vals[cd]:
                        retvals[cd].append(check_data)

    return retvals


def return_borough_tags(valid_authors, in_file):
    """
    given the shapefile of boroughs and an input file plus valid authors...
    we return a series of objects: classification + tags
    :param valid_authors:
    :param in_file:
    :return:
    """
    starttime = time.time()
    # not_nyc = fiona.open("./shp/empty_nyc.shp", 'r')
    # for poly in not_nyc:
    #     not_nyc_shp = shape(poly['geometry'])

    boroughs = fiona.open('./shp/nybb_wgs84.shp', 'r')
    geoms = list()
    for item in boroughs:
        shply_shape = shape(item['geometry'])
        adder = [shply_shape, item['properties']['BoroName']]
        geoms.append(adder)

    swfile = file("./stopwords.txt", 'r')
    stopwords = list()
    for ln in swfile:
        stopwords.append(ln.lower().strip())
    swfile.close()

    in_reader = open(in_file, 'r')
    tag_returns = list()
    count = 0
    totcount = 0
    for line in in_reader:
        try:
            totcount += 1
            jsline = json.loads(line)
            if jsline['owner'] in valid_authors:
                b_class = get_borough_item(jsline, geoms, stopwords)
                # Get the borough class from this function.
                if b_class >= 0:
                    tag_returns.append(b_class)

        except ValueError:
            pass
    print count, totcount
    print len(tag_returns)
    # print tag_returns
    tot_time = time.time() - starttime
    print tot_time
    return tag_returns


def connect():
    """
    Connect to DB.
    :return:
    """
    global CONNECTION
    conn_str = "dbname='jimmy1' user='jimmy1' " \
               "host='localhost' " \
               "port='5432' "
    CONNECTION = psycopg2.connect(conn_str)
    return CONNECTION


def import_line(line):
    """
    Given a line, put it in the database (if it's not already there)
    :param line: an object, formerly from JSON
    :return:
    """
    curs = CONNECTION.cursor()

    sql = "SELECT COUNT(*) FROM flickr_data WHERE id=%s"
    data = (line['id'],)
    curs.execute(sql, data)
    res = curs.fetchone()

    if res[0] == 0:
        flds = ['ispublic', 'place_id', 'geo_is_public', 'owner',
                'id', 'title', 'woeid', 'geo_is_friend',
                'geo_is_contact', 'datetaken', 'isfriend', 'secret',
                'ownername', 'latitude', 'longitude', 'accuracy',
                'isfamily', 'tags', 'farm', 'geo_is_family',
                'dateupload', 'datetakengranularity', 'server', 'context']
        fields = list()
        data = list()
        for fld in flds:
            if fld in line:
                fields.append(fld)
                data.append(line[fld])
        fields = ",".join(fields)
        dataholder = (",%s"*len(data))[1:]
        sql = "INSERT INTO flickr_data (" + fields + ") VALUES (" + dataholder + ")"

        try:
            curs.execute(sql, data)
            CONNECTION.commit()
        except psycopg2.Error, err:
            print("ERROR: {0} {1}".format(
                err.diag.severity, err.pgerror))
    else:
        return False
    return line['id']


def file_to_database(filename):
    """
    take a file and push all JSON to database
    :param filename:
    :return:
    """
    print filename
    global CONNECTION
    CONNECTION = connect()

    infile = open(filename, 'r')
    count = 0
    ids = set()
    inserts = 0
    for line in infile:
        json_line = None
        try:
            json_line = json.loads(line)
        except:
            json_line = None
            continue
        if json_line is not None:
            new_id = import_line(json_line)
            count += 1
            if new_id is not False:
                inserts += 1

    print "Imported: ", inserts, " out of ", count


def train_dataset(ret_tags):

    ## Below lines of code find the authors that are valid for
    ## inclusion in our training set.
    # tgt_nyc_file = "nyc_json.json"
    # nyc_authors = multi_day_authors(tgt_nyc_file, 5)
    # print "Authors: ", len(nyc_authors)
    # pickle.dump(nyc_authors, file("nyc_authors.pickle", 'w'))
    # ret_tags = return_borough_tags(nyc_authors, tgt_nyc_file)
    # pickle.dump(ret_tags, file("nyc_ret_tags.pickle", 'w'))
    # print "Tag Count: ", len(ret_tags)
    # exit(-1)
    random.shuffle(ret_tags)
    # ret_tags = pickle.load(file('nyc_ret_tags.pickle', 'r'))
    nyc_authors = pickle.load(file('nyc_authors.pickle', 'r'))
    boroughs = [u'Staten Island', u'Manhattan', u'Bronx', u'Brooklyn', u'Queens']

    print ret_tags[0]

    line = int(math.floor(len(ret_tags) * .80))
    print line, len(ret_tags)

    all_vals = np.array([_[0] for _ in ret_tags])
    print "All Values: ", np.histogram(all_vals,
                                       bins=[-.5, 0.5, 1.5, 2.5, 3.5, 4.5])

    trng_vals = [_[0] for _ in ret_tags[0:line]]
    trng_docs = [_[1] for _ in ret_tags[0:line]]

    test_vals = [_[0] for _ in ret_tags[line:]]
    test_docs = [_[1] for _ in ret_tags[line:]]

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer(use_idf=False, norm='l1')),
                         # ('clf', MultinomialNB())
                         ('clf', SGDClassifier(penalty='elasticnet',
                                               alpha=1e-06, n_iter=50))
                        ])
    # MultinomialNB()

    # Code below is for a parameter search over the pipeline:

    # parameters = {
    #     'vect__max_df': (0.5, 0.75, 1.0),
    #     # 'vect__max_features': (None, 5000, 10000, 50000),
    #     'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #     'tfidf__use_idf': (True, False),
    #     'tfidf__norm': ('l1', 'l2'),
    #     'clf__alpha': (0.00001, 0.000001),
    #     'clf__penalty': ('l2', 'elasticnet'),
    #     'clf__n_iter': (10, 50, 80),
    # }
    # grid_search = GridSearchCV(text_clf, parameters, n_jobs=-1, verbose=1)
    #
    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in text_clf.steps])
    # print("parameters:")
    # print(parameters)
    # t0 = time.time()
    # grid_search.fit(trng_docs, trng_vals)
    # print("done in %0.3fs" % (time.time() - t0))
    # print()
    #
    # print("Best score: %0.3f" % grid_search.best_score_)
    # print("Best parameters set:")
    # best_parameters = grid_search.best_estimator_.get_params()
    # for param_name in sorted(parameters.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))

    text_clf = text_clf.fit(trng_docs[:line], trng_vals[:line])

    predicted = text_clf.predict(test_docs)

    print np.histogram(predicted, bins=[-.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    print "Average prediction: ", np.mean(predicted == test_vals)

    return text_clf


def pickle_training_set(vals, outfile):
    connect()
    thresholds = [vals for _ in range(0, 5)]
    print thresholds
    ret_vals = collect_nyc_training_data(thresholds)
    outfile = file(outfile, 'w')
    pickle.dump(ret_vals, outfile)
    outfile.close()


if __name__ == "__main__":
    # train_dataset()
    # pickle_training_set(1000, "./nyc_trng_100.pickle")

    nyc_trng_file = file("./nyc_trng_100.pickle", 'r')
    nyc_trng = pickle.load(nyc_trng_file)
    nyc_trng_file.close()

    trng_vals = list()
    for i in range(0, len(nyc_trng)):
        for j in range(0, len(nyc_trng[i])):
            trng_vals.append(nyc_trng[i][j])

    classifier = train_dataset(trng_vals)

    



