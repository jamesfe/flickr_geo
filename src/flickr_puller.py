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
import psycopg2
import os

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

    tags = json_line['tags'].lower()
    if ret_val[0] < 0:
        return -1
    elif len(tags) > 5:  # more than 5 chars
        for sw in stopwords:
            tags = tags.replace(sw, "")
            json_line['tags'] = tags
        ret_val.append(json_line)
    return ret_val


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


def build_targets(inpt, num, basename):
    latdiff = inpt[0][0] - inpt[1][0]
    londiff = inpt[0][1] - inpt[1][1]
    retvals = list()
    for latval in range(0, num):
        for lonval in range(0, num):
            newlat = inpt[0][0] + (latdiff / num) * latval
            newlon = inpt[0][1] + (londiff / num) * lonval
            fname = basename + "_" + str(latval) + "_" + str(lonval) + ".json"
            r = dict({'lat': newlat, 'lon': newlon, 'fname': fname})
            retvals.append(r)
    return retvals


def pull_data(minpage, minday):
    """
    generic function to grab more data
    :return:
    """
    dc_pt = [[40, -78], [37, -75]]
    ny_pt = [[42, -75], [39, -73]]

    targets = build_targets(dc_pt, 10, "dc_targets")
    targets.extend(build_targets(ny_pt, 10, "nyc_targets"))

    for target in targets:
        tm = "./fix_megapull_2012/" + str(time.time()).split(".")[0]
        target['file'] = open(tm + target['fname'], 'a')

    totcount = 0
    for curr_eval in range(minpage, 10):
        print "Page Pull: " + str(curr_eval)
        base_time = 1325379661
        if curr_eval == minpage:
            md = minday
        else:
            md = 0

        for sec_time in range(md, 365):
            yearcount = 0
            dt = base_time + (sec_time * 86400)
            print time.asctime(), "Day of Year: ", sec_time, " Photos: ", \
                yearcount, totcount
            for target in targets:
                ret_photos = get_payload(target['lat'], target['lon'],
                                         curr_eval, dt)
                for photo in ret_photos:
                    target['file'].write(json.dumps(photo) + "\n")
                    yearcount += 1
                    totcount += 1
            print yearcount, totcount

    for target in targets:
        target['file'].close()


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


def file_to_database(filename):
    """
    take a file and push all JSON to database
    :param filename:
    :return:
    """
    print filename
    global CONNECTION
    CONNECTION = connect()
    curs = CONNECTION.cursor()
    infile = open(filename, 'r')
    count = 0
    ids = set()
    inserts = 0
    for line in infile:
        # if count > 100:
        #     break
        count += 1
        if count % 10000 == 0:
            print "Count: ", count, inserts
        try:
            line = json.loads(line)
        except:
            continue

        res = list([0])
        if line['id'] not in ids:
            sql = "SELECT COUNT(*) FROM flickr_data WHERE id=%s"
            data = (line['id'],)
            curs.execute(sql, data)
            res = curs.fetchone()
        else:
            res = list([1])

        if res[0] >= 1:
            ids.add(line['id'])

        if res[0] == 0:
            # print line
            flds = ['ispublic',
                    'place_id',
                    'geo_is_public',
                    'owner',
                    'id',
                    'title',
                    'woeid',
                    'geo_is_friend',
                    'geo_is_contact',
                    'datetaken',
                    'isfriend',
                    'secret',
                    'ownername',
                    'latitude',
                    'longitude',
                    'accuracy',
                    'isfamily',
                    'tags',
                    'farm',
                    'geo_is_family',
                    'dateupload',
                    'datetakengranularity',
                    'server',
                    'context']
            fields = list()
            data = list()
            for fld in flds:
                if fld in line:
                    fields.append(fld)
                    data.append(line[fld])
            fields = ",".join(fields)
            dataholder = (",%s"*len(data))[1:]
            sql = "INSERT INTO flickr_data (" + fields + ") VALUES (" + dataholder + ")"
            # print curs.mogrify(sql, data)

            try:
                curs.execute(sql, data)
                inserts += 1
                CONNECTION.commit()
                ids.add(line['id'])
            except psycopg2.Error, err:
                print("ERROR: {0} {1}".format(
                    err.diag.severity, err.pgerror))

    print "Imported: ", inserts, " out of ", count

if __name__ == '__main__':
    pull_data(1, 1)
    #
    # for k in os.listdir("./megapull2012/"):
    #     file_to_database("./megapull2012/" + k)


def train_dataset():

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

    ret_tags = pickle.load(file('nyc_ret_tags.pickle', 'r'))
    nyc_authors = pickle.load(file('nyc_authors.pickle', 'r'))
    boroughs = [u'Staten Island', u'Manhattan', u'Bronx', u'Brooklyn', u'Queens']

    line = int(math.floor(len(ret_tags) * .80))
    print line, len(ret_tags)

    all_vals = np.array([_[0] for _ in ret_tags])
    print "All Values: ", np.histogram(all_vals, bins=[-.5, 0.5, 1.5, 2.5, 3.5, 4.5])

    target_vals = [_[0] for _ in ret_tags[0:line]]
    trng_docs = [_[1] for _ in ret_tags[0:line]]

    test_vals = [_[0] for _ in ret_tags[line:]]
    test_docs = [_[1] for _ in ret_tags[line:]]

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5))])
    # MultinomialNB()
    text_clf = text_clf.fit(trng_docs[:line], target_vals[:line])

    predicted = text_clf.predict(test_docs)
    print np.histogram(predicted, bins=[-.5, 0.5, 1.5, 2.5, 3.5, 4.5])

    print np.mean(predicted == test_vals)

    # predicted = text_clf.predict(trng_docs[line:])
    # print predicted.shape
    # print np.mean(predicted == target_vals[line:len(ret_tags)])
    #
    # Predict some stuff and visually check it
    # predicted = text_clf.predict(trng_docs[0:50])
    # print np.mean(predicted == target_vals[0:50])
    # for i in zip(predicted, trng_docs):
    #     print i



    #
    # parameters = {
    #     'vect__max_df': (0.5, 0.75, 1.0),
    #     #'vect__max_features': (None, 5000, 10000, 50000),
    #     'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #     #'tfidf__use_idf': (True, False),
    #     #'tfidf__norm': ('l1', 'l2'),
    #     'clf__alpha': (0.00001, 0.000001),
    #     'clf__penalty': ('l2', 'elasticnet'),
    #     #'clf__n_iter': (10, 50, 80),
    # }
    # grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    # begin the learning process:



    # pull_data()
    # print nyc_authors

    # pic_count = 0
    # tot_count = 0
    # for k in open(tgt_nyc_file, 'r'):
    #     r = json.loads(k)
    #     tot_count += 1
    #     if r['owner'] in nyc_authors:
    #         pic_count += 1
    # print pic_count, tot_count



    print "done"

    # [1, ["nyc", "boring", "hipster"]]

    ## sample
    # {"ispublic": 1, "place_id": "5ebRtgVTUro3guGorw", "geo_is_public": 1,
    # "owner": "72098626@N00", "id": "15604638529", "title": "Skateboard kid #8",
    # "woeid": "20070197", "geo_is_friend": 0, "geo_is_contact": 0, "datetaken":
    # "2014-09-28 16:02:01", "isfriend": 0, "secret": "abf65e17a8", "ownername":
    # "Ed Yourdon", "latitude": 40.731371, "accuracy": "16", "isfamily": 0,
    # "tags": "newyork greenwichvillage streetsofnewyork streetsofny everyblock",
    # "farm": 8, "geo_is_family": 0, "dateupload": "1416009675",
    # # "datetakengranularity": "0", "longitude": 74.005447, "server": "7477", "context": 0}

