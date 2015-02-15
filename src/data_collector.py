"""
A program to pull data over a set of coordinates  Uses flickr_puller
"""

import time
import os
import json
import pickle
import threading
import datetime as dt
import psycopg2
import requests

API_KEY = open("api_key.txt", 'r').read().strip()


def get_payload(city_lat, city_lon, c_page, min_date=None):
    """
    Get a number of photos within 32km of a point.
    :param city_lat: latitude in DD
    :param city_lon: longitude in DD
    :param c_page:  which page do you want? 1-n
    :return:
    """
    start = time.time()
    threshold = 18  # number of seconds we want a call to take
    payload = {"method": "flickr.photos.search",
               "accuracy": 8,
               "lat": city_lat,
               "lon": city_lon,
               "radius": 30,
               "extras": "date_upload,date_taken,owner_name,geo,tags",
               "per_page": 500,
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


class DataCollector(threading.Thread):
    def __init__(self, checker_name, basepath, basename, top_left,
                 bottom_right, numpieces, start_date):
        """
        Initializer.
        :param checker_name: filename for a .pickle file to check progress
        :param basepath: base path for the storage of JSON backups
        :param basename: base name for the JSON backup files
        :param top_left: upper left corner of collection area [lat, lon]
        :param bottom_right: bottom right corner of coll area [lat, lon]
        :param numpieces: number of pieces to divide area into
        :param start_date: datetime object to start collections for
        that year on
        :return:
        """
        threading.Thread.__init__(self)
        self.checker = checker_name
        self.basename = basename
        self.tgt_list = build_targets([top_left, bottom_right],
                                      numpieces, basename)
        self.stime = int(time.mktime(start_date.timetuple()))
        self.basepath = basepath

        self.numpieces = numpieces

        if not os.path.isdir(basepath):
            os.mkdir(basepath)
            print "Made directory: ", basepath

        for target in self.tgt_list:
            tm = os.path.join(basepath,
                              str(time.time()).split(".")[0] + target['fname'])
            target['file'] = open(tm, 'a')

        conn_str = "dbname='jimmy1' user='jimmy1' " \
                   "host='localhost' " \
                   "port='5432' "
        self.conn = psycopg2.connect(conn_str)

        sql = "SELECT id FROM flickr_data"
        curs = self.conn.cursor()
        curs.execute(sql)
        self.ids = set()
        results = curs.fetchall()
        for res in results:
            self.ids.add(res[0])

    def run(self):
        """
        run the collector
        :return:
        """
        print "Starting.", self.checker
        self.pull_data()

    def pull_data(self):
        """
        generic function to grab more data
        :return:
        """
        start_checker = get_checker(self.checker)
        md = start_checker['sec_time']

        for sec_time in range(md, 365):
            print time.asctime(), self.basename, " seconds: ", sec_time
            minpage = 0
            if sec_time != start_checker['sec_time']:
                    minpage = start_checker['curr_page']

            for curr_eval in range(minpage, 5):  # number of pages left = 5

                checker = file(self.checker, 'w')
                pickle.dump(dict({"sec_time": sec_time,
                                  "curr_page": curr_eval}), checker)
                checker.close()
                delta_time = self.stime + (sec_time * 86400)

                for index, target in enumerate(self.tgt_list):
                    if target['dead'] is not True:
                        ret_photos = get_payload(target['lat'], target['lon'],
                                                 curr_eval, delta_time)
                        numinserted = 0
                        for photo in ret_photos:
                            target['file'].write(json.dumps(photo) + "\n")
                            r = self.import_line(photo)
                            if r is not False:
                                numinserted += 1
                        if len(ret_photos) == 0 or numinserted == 0:
                            self.tgt_list[index]['dead'] = True
                        elif numinserted > 0:
                            print time.asctime(), self.basename, " received: ", \
                                len(ret_photos), " Inserted: ", numinserted

            for index, target in enumerate(self.tgt_list):
                self.tgt_list[index]['dead'] = False

        for target in self.tgt_list:
            target['file'].close()

    def import_line(self, line):
        """
        Given a line, put it in the database (if it's not already there)
        :param line: an object, formerly from JSON
        :return:
        """
        curs = self.conn.cursor()

        if line['id'] in self.ids:
            return False

        sql = "SELECT COUNT(*) FROM flickr_data WHERE id=%s"
        data = (line['id'],)
        curs.execute(sql, data)
        res = curs.fetchone()
        self.ids.add(line['id'])

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
            sql = "INSERT INTO flickr_data (" \
                  + fields + ") VALUES (" + dataholder + ")"

            try:
                curs.execute(sql, data)
                self.conn.commit()
            except psycopg2.Error, err:
                print("ERROR: {0} {1}".format(
                    err.diag.severity, err.pgerror))
        else:
            return False
        return line['id']


def build_targets(inpt, num, basename):
    """
    build a grid of lat/longs that tell us where we will be collecting.
    :param inpt:
    :param num:
    :param basename:
    :return:
    """
    lats = sorted(list([inpt[0][0], inpt[1][0]]))
    lons = sorted(list([inpt[0][1], inpt[1][1]]))

    maxlat = max(lats)
    minlat = min(lats)
    maxlon = max(lons)
    minlon = min(lons)
    latdiff = maxlat - minlat
    londiff = maxlon - minlon
    retvals = list()
    for latval in range(0, num):
        for lonval in range(0, num):
            newlat = minlat + (latdiff / float(num)) * latval
            newlon = minlon + (londiff / float(num)) * lonval
            fname = basename + "_" + str(latval) + "_" + str(lonval) + ".json"
            r = dict({'lat': newlat, 'lon': newlon,
                      'fname': fname, 'dead': False})
            retvals.append(r)
    return retvals


def get_checker(fname):
    """
    get where we last left off
    :param fname:
    :return:
    """
    try:
        r = file(fname, 'r')
    except:
        return dict({"sec_time": 1, "curr_page": 1})
    check = pickle.load(r)
    print check
    return check


if __name__ == "__main__":

    colls = list()

    # for yr in [2011, 2012, 2013, 2014]:
    for yr in [2006, 2007, 2008, 2009, 2010]:
        yrs = str(yr)
        colls.append(DataCollector("./checkers/hampton_roads"+yrs+".pickle",
                                   "./data/hroads"+yrs, "hr_data"+yrs,
                                   [37.91, -77.81], [35.8, -75.40],
                                   10, dt.datetime(yr, 1, 1)))
        colls.append(DataCollector("./checkers/sf_checker"+yrs+".pickle",
                                   "./data/sfo"+yrs, "sfo_tgts_test"+yrs,
                                   [38, -123], [37, -120],
                                   5, dt.datetime(yr, 1, 1)))
        colls.append(DataCollector("./checkers/nyc"+yrs+".pickle",
                                   "./data/nyc/"+yrs, "nyc"+yrs,
                                   [42, -75], [40, -72],
                                   10, dt.datetime(yr, 1, 1)))
        colls.append(DataCollector("./checkers/bigdc"+yrs+".pickle",
                                   "./data/wdc/"+yrs, "wdc"+yrs,
                                   [40, -78], [37.5, -76],
                                   8, dt.datetime(yr, 1, 1)))
        colls.append(DataCollector("./stisle"+yrs+".pickle",
                                   "./data/stislestisle"+yrs, "stl"+yrs,
                                   [40.6, -74.3], [40.4, -73.9],
                                   3, dt.datetime(yr, 1, 1)))

    for i in colls:
        i.start()
