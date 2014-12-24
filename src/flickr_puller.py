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
from sklearn.pipeline import Pipeline
import numpy as np
import math

from geohash import encode, bbox

API_KEY = open("api_key.txt", 'r').read().strip()
NY_LAT = 40.69
NY_LON = -74.00
DC_LAT = 38.90
DC_LON = -77.00

BRONX_LON = -73.86
BRONX_LAT = 40.85
STISL_LON = -74.14
STISL_LAT = 40.58


def get_payload(city_lat, city_lon, c_page, min_date=None):
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
               "radius": 30,
               "extras": "date_upload,date_taken,owner_name,geo,tags",
               "per_page": 250,
               "format": "json",
               "nojsoncallback": 1,
               "page": c_page,
               "api_key": API_KEY}
    if isinstance(min_date, int):
        payload["min_upload_date"] = min_date
        payload["max_upload_date"] = min_date+86400

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
        tags = clean_tags(json_line['tags'], stopwords)

        if len(tags) > 20:  # more than 20 chars
            ret_val.append(tags)
            return ret_val
    else:
        return False


def collect_nyc_training_data(thr_vals, look_box=None):
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
    num_pull = len(thr_vals[0]) * 10

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
        if look_box is not None:
            sql = "SELECT id, internal_id, tags, latitude, longitude " \
                  "FROM flickr_data WHERE internal_id > %s " \
                  "AND latitude < %s " \
                  "AND latitude > %s " \
                  "AND longitude < %s" \
                  "AND longitude > %s" \
                  "ORDER BY internal_id ASC LIMIT %s"
        data = (curr_id, look_box['top'], look_box['bottom'],
                look_box['left'], look_box['right'], num_pull)
        curs.execute(sql, data)
        res = curs.fetchall()
        if len(res) == 0:
            print "Not enough data."
            return retvals
        for r in res:
            curr_id = r['internal_id']
            if r['latitude'] > 40:
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
    random.shuffle(ret_tags)
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

    text_clf = text_clf.fit(trng_docs[:line], trng_vals[:line])

    predicted = text_clf.predict(test_docs)

    print np.histogram(predicted, bins=[-.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    print "Average prediction: ", np.mean(predicted == test_vals)

    return text_clf


def pickle_training_set(vals, outfile):
    connect()
    thresholds = [vals for _ in range(0, 5)]
    print thresholds
    look_box = dict({
        "top": 42,
        "bottom": 40,
        "left": -74,
        "right": -72})

    ret_vals = collect_nyc_training_data(thresholds, look_box)
    outfile = file(outfile, 'w')
    pickle.dump(ret_vals, outfile)
    outfile.close()


def clean_tags(in_string, stopwords):
    """
    take an instring
    tokenize
    clean with stopwords
    return string
    :param in_string:
    :return:
    """
    tags = in_string.lower()
    # Tokenize, deduplicate, strip, restring - all one step.
    tags = set([_ for _ in tags.split(" ") if len(_.strip()) > 0])
    ret_val = list()
    for tag in tags:
        if tag not in stopwords and tag.find("foursquare") == -1:
            ret_val.append(tag)
    return ' '.join(ret_val)


def classify_database(clsfy, classrun, notes, num):
    """
    run this classifier and throw some notes in the database -
    :param clsfy: scikit learn classifier (trained)
    :param classrun: a number, preferably uniqueish
    :param notes: notes about the classifier
    :param num: number of records to do (-1 for all)
    :return:
    """
    connect()
    curs = CONNECTION.cursor()

    swfile = file("./stopwords.txt", 'r')
    stopwords = list()
    for ln in swfile:
        stopwords.append(ln.lower().strip())
    swfile.close()

    if num < 0:
        num = " LIMIT ALL"
    else:
        num = " LIMIT " + str(num)

    sql = "SELECT internal_id, latitude, longitude, tags FROM flickr_data" + num
    curs.execute(sql)
    res = curs.fetchall()

    count = 0
    for r in res:
        count += 1
        if (count % 1000) == 0:
            print time.asctime(), count
        tags = r[3]
        tags = clean_tags(tags, stopwords)
        prediction = int(clsfy.predict([r[3]])[0]) # this might be an issue
        sql = "INSERT INTO classifications " \
              "(pred_code, fl_internal_id, notes, " \
              "classrun, latitude, longitude) " \
              "VALUES (%s, %s, %s, %s, %s, %s)"
        data = (prediction, r[0], notes, classrun, r[1], r[2])
        try:
            curs.execute(sql, data)
            CONNECTION.commit()
        except Exception as err:
            print curs.mogrify(sql, data)
            print "Exception of type : " + str(type(err)) \
                  + ".  Rolling back..." + str(err)
            ## pick the data, then add to another DB row


def geohash_to_polygons(classrun, num, accuracy=8):
    """
    turn a class run into a JSON file for leaflet to interpret
    :param classrun:
    :return:
    """
    connect()
    sql = "SELECT pred_code, latitude, longitude " \
          "FROM classifications WHERE classrun=%s LIMIT %s"
    data = (classrun, num)
    curs = CONNECTION.cursor()
    curs.execute(sql, data)
    results = curs.fetchall()

    geohashes = dict()
    for res in results:
        lat = res[1]
        lon = res[2]
        code = res[0]
        geo_hsh = encode(lat, lon, accuracy)
        if geo_hsh in geohashes:
            if code in geohashes[geo_hsh]:
                geohashes[geo_hsh][code] += 1
            else:
                geohashes[geo_hsh][code] = 1
        else:
            geohashes[geo_hsh] = dict()
            geohashes[geo_hsh][code] = 1

    geojson_obj = list()

    for res in geohashes:
        new_obj = dict({"type": "Feature",
                        "id": res,
                        "geometry": dict(),
                        "properties": dict()})
        val = max(geohashes[res])
        new_obj['properties']['pred_code'] = val
        poly = list()
        ghbox = bbox(res)
        poly.append([ghbox['w'], ghbox['n']])
        poly.append([ghbox['w'], ghbox['s']])
        poly.append([ghbox['e'], ghbox['s']])
        poly.append([ghbox['e'], ghbox['n']])
        new_obj['geometry']['coordinates'] = [poly]
        new_obj['geometry']['type'] = "Polygon"
        geojson_obj.append(new_obj)

    ret_vals = dict({"type": "FeatureCollection",
                     "features": geojson_obj})

    return ret_vals


def train_to_database():
    """
    train the classifier, then do a bunch of values

    :return:
    """
    nyc_trng_file = file("./nyc_trng_100.pickle", 'r')
    nyc_trng = pickle.load(nyc_trng_file)
    nyc_trng_file.close()

    trng_vals = list()
    for i in range(0, len(nyc_trng)):
        print "Row ", i, ": ", len(nyc_trng[i])
        for j in range(0, len(nyc_trng[i])):
            trng_vals.append(nyc_trng[i][j])

    classifier = train_dataset(trng_vals)

    classify_database(classifier, 3, "3000 data points each", 550000)


def find_overlapping_tags():
    nyc_trng_file = file("./nyc_trng_100.pickle", 'r')
    nyc_trng = pickle.load(nyc_trng_file)
    nyc_trng_file.close()

    trng_vals = list()

    for i in range(0, len(nyc_trng)):
        print "Row ", i, ": ", len(nyc_trng[i])
        for j in range(0, len(nyc_trng[i])):
            trng_vals.append(nyc_trng[i][j])

    classifier = train_dataset(trng_vals)
    stopwords = get_stopwords()

    for i in trng_vals:
        in_tags = [clean_tags(i[1], stopwords)]
        pred = classifier.predict(in_tags)[0]
        if pred == 1 and i[0] != 1:
            print in_tags[0],


def get_stopwords():
    swfile = file("./stopwords.txt", 'r')
    stopwords = list()
    for ln in swfile:
        stopwords.append(ln.lower().strip())
    swfile.close()
    return stopwords


if __name__ == "__main__":
    # pickle_training_set(2000, "./nyc_trng_10000.pickle")
    # train_to_database()
    # print "Done classifying, outputting... "

    # ofile = file("./webapp/data/outfile_geohash6.json", 'w')
    # rvals = geohash_to_polygons(1, 550000, 6)
    # ofile.write("var inData =")
    # ofile.write(json.dumps(rvals))
    # ofile.write(";\n")
    # ofile.close()
    # print "done"

    find_overlapping_tags()