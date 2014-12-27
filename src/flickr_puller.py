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
import psycopg2.extras
import random

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import math

from geohash import encode, bbox
from operator import itemgetter


NY_LAT = 40.69
NY_LON = -74.00
DC_LAT = 38.90
DC_LON = -77.00

BRONX_LON = -73.86
BRONX_LAT = 40.85
STISL_LON = -74.14
STISL_LAT = 40.58


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


def get_borough_item(json_line, borough_geoms, stopwords, findwords):
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
        tags = json_line['tags'] + " " + json_line['title']
        tags = clean_tags(tags, stopwords, findwords)

        if len(tags) > 20:  # more than 20 chars
            ret_val.append(tags)
            return ret_val
    else:
        return False


def gather_nyc_data(thr_vals, class_notes, outfile):
    """
    A function that takes an array like [10,10,10,10,10] and returns
    that many values (10 of each) for each index in geo_class
    :param thr_vals:
    :return:
    """
    connect()
    curs = CONNECTION.cursor(cursor_factory=psycopg2.extras.DictCursor)
    ret_vals = list([[] for _ in thr_vals])

    stopwords = get_stopwords()
    findwords = get_findwords()

    for index, thr in enumerate(thr_vals):
        sql = "SELECT geo_code, tags, title FROM flickr_data fd " \
              "INNER JOIN geo_class gc " \
              "ON fd.internal_id=gc.internal_id " \
              "WHERE gc.geo_code=%s " \
              "AND gc.geo_notes=%s LIMIT %s"
        data = (index, class_notes, thr * 2)
        print curs.mogrify(sql, data)
        curs.execute(sql, data)
        res = curs.fetchall()
        for r in res:
            tags = r['tags'] + " " + r['title']
            tags = clean_tags(tags, stopwords, findwords)
            if len(tags) > 0 and len(ret_vals[index]) < thr_vals[index]:
                ret_vals[index].append([r['geo_code'], tags])

    print [len(_) for _ in ret_vals]

    outfile = file(outfile, 'w')
    pickle.dump(ret_vals, outfile)
    outfile.close()


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
        sql = "INSERT INTO flickr_data (" + fields + \
              ") VALUES (" + dataholder + ")"

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
    print "Sample: "
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
                         ('clf', SGDClassifier(penalty='elasticnet',
                                               alpha=1e-06, n_iter=50))])

    text_clf = text_clf.fit(trng_docs[:line], trng_vals[:line])

    predicted = text_clf.predict(test_docs)

    print np.histogram(predicted, bins=[-.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    print "Average prediction: ", np.mean(predicted == test_vals)

    return text_clf


def clean_tags(in_string, stopwords, findwords):
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
        tag = tag.replace("#", "")
        # print tag.find("uploaded")
        if tag not in stopwords:
            found = False
            for item in findwords:
                if tag.find(item) > -1:
                    found = True
            if not found:
                if len(tag) > 2:
                    ret_val.append(tag)
                # print "Adding: ", tag, tag.find(item), item
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

    stopwords = get_stopwords()
    findwords = get_findwords()

    if num < 0:
        num = " LIMIT ALL"
    else:
        num = " LIMIT " + str(num)

    sql = "SELECT internal_id, latitude, longitude, tags, title " \
          "FROM flickr_data" + num
    curs.execute(sql)
    res = curs.fetchall()

    count = 0
    for r in res:
        count += 1
        if (count % 10000) == 0:
            print time.asctime(), count
        tags = r[3] + " " + r[4]
        tags = clean_tags(tags, stopwords, findwords)
        prediction = int(clsfy.predict([tags])[0])  # this might be an issue
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
            # pick the data, then add to another DB row


def geohash_to_polygons(classrun, num, look_box, accuracy=8):
    """
    turn a class run into a JSON file for leaflet to interpret
    :param classrun:
    :return:
    """
    connect()
    lats = sorted(look_box['lats'])
    lons = sorted(look_box['lons'])

    sql = "SELECT pred_code, latitude, longitude " \
          "FROM classifications WHERE classrun=%s " \
          "AND latitude < %s " \
          "AND latitude > %s " \
          "AND longitude < %s " \
          "AND longitude > %s " \
          "LIMIT %s"
    data = (classrun, lats[1], lats[0], lons[1], lons[0], num)
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
        # val = max(geohashes[res])

        val = sorted(geohashes[res].items(),
                     key=itemgetter(1), reverse=True)[0][0]

        # print val, geohashes[res]
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


def train_to_database(num_trng, notes, classrun, label_datafile):
    """
    train the classifier, then do a bunch of values

    :return:
    """
    nyc_trng_file = file(label_datafile, 'r')
    nyc_trng = pickle.load(nyc_trng_file)
    nyc_trng_file.close()

    trng_vals = list()
    for i in range(0, len(nyc_trng)):
        print "Row ", i, ": ", len(nyc_trng[i])
        for j in range(0, len(nyc_trng[i])):
            trng_vals.append(nyc_trng[i][j])

    classifier = train_dataset(trng_vals)

    classify_database(classifier, classrun, notes, num_trng)


def find_overlapping_tags(trng_file, over_file):
    nyc_trng_file = file(trng_file, 'r')
    nyc_trng = pickle.load(nyc_trng_file)
    nyc_trng_file.close()

    trng_vals = list()

    for i in range(0, len(nyc_trng)):
        print "Row ", i, ": ", len(nyc_trng[i])
        for j in range(0, len(nyc_trng[i])):
            trng_vals.append(nyc_trng[i][j])

    classifier = train_dataset(trng_vals)
    stopwords = get_stopwords()
    findwords = get_findwords()

    ofile = file(over_file, 'w')

    for i in trng_vals:
        in_tags = [clean_tags(i[1], stopwords, findwords)]
        pred = classifier.predict(in_tags)[0]
        if i[0] != pred:
            ofile.write(i[1] + "\n")

    ofile.close()


def get_stopwords():
    swfile = file("./stopwords.txt", 'r')
    stopwords = list()
    for ln in swfile:
        stopwords.append(ln.lower().strip())
    swfile.close()
    return stopwords


def get_findwords():
    swfile = file("./findwords.txt", 'r')
    stopwords = list()
    for ln in swfile:
        stopwords.append(ln.lower().strip())
    swfile.close()
    return stopwords


def getwords(tgt):
    """
    get a pile of words for analysis
    :param tgt:
    :return:
    """
    sw = get_stopwords()
    fw = get_findwords()

    nyc_trng_file = file("./nyc_trng_100.pickle", 'r')
    nyc_trng = pickle.load(nyc_trng_file)
    nyc_trng_file.close()

    wordlist = ' '
    for i in nyc_trng[tgt]:
        tags = clean_tags(i[1], sw, fw)
        wordlist += tags
    print wordlist


def perform_geo_class_nyc():
    """
    since the slowest part of this is figuring out which borough things are in,
    we are going to classify them and store the result in SQL where a JOIN
    will hopefully be quicker.
    :return:
    """
    connect()
    boroughs = fiona.open('./shp/nybb_wgs84.shp', 'r')
    geoms = list()
    for item in boroughs:
        shply_shape = shape(item['geometry'])
        adder = [shply_shape, item['properties']['BoroName']]
        geoms.append(adder)

    curs = CONNECTION.cursor(cursor_factory=psycopg2.extras.DictCursor)
    notes = "nyc_class"

    sql = "SELECT fd.latitude, fd.longitude, fd.internal_id, gc.geo_notes " \
          "FROM flickr_data fd " \
          "LEFT JOIN geo_class gc ON (fd.internal_id=gc.internal_id) " \
          "WHERE gc.internal_id IS NULL"
    # Warning: doens't take into account non-NYC classifications
    curs.execute(sql)
    res = curs.fetchall()
    print len(res)
    for numcoord, line in enumerate(res):
        if (numcoord % 10000) == 0:
            print time.asctime(), "Geo categorized: ", numcoord
        tgt_point = Point(line['longitude'], line['latitude'])
        # if not tgt_point.intersects(not_nyc_shp):
        tval = -1
        for index, geo in enumerate(geoms):
            if tgt_point.intersects(geo[0]):
                tval = index
                break

        geo_code = tval
        if tval > -1:
            geo_text = geoms[tval][1]
        else:
            geo_text = "none"
        sql = "INSERT INTO geo_class" \
              " (geo_code, geo_text, internal_id, geo_notes) " \
              "VALUES (%s, %s, %s, %s)"
        data = (geo_code, geo_text, line['internal_id'], notes)
        curs.execute(sql, data)
        CONNECTION.commit()

if __name__ == "__main__":
    perform_geo_class_nyc()
    trn_file = "./pickles/out_7000.pickle"
    gather_nyc_data([15000]*5, 'nyc_class', trn_file)
    print "Finding Overlaps: "
    find_overlapping_tags(trn_file, "overlaps.txt")
    print "Overlaps complete."
    train_to_database(-1, "total_redo", 7001, trn_file)
    hashlen = 5
    ofile = file("./webapp/data/dc_outfile_geohash"+str(hashlen)+".json", 'w')
    dc_lbox = dict({"lats": [40, 37.5], "lons": [-78, -76]})
    # nyc_lbox = dict({"lats": [42, 40], "lons": [-75, -72]})
    rvals = geohash_to_polygons(7001, 1000000, dc_lbox, hashlen)
    ofile.write("var inData =")
    ofile.write(json.dumps(rvals))
    ofile.write(";\n")
    ofile.close()
    print "done"
