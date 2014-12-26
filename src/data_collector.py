"""
A program to pull data over a set of coordinates  Uses flickr_puller
"""

import time
import os
import json
import pickle
from flickr_puller import get_payload
import threading
import datetime as dt
import psycopg2


class DataCollector(threading.Thread):
    def __init__(self, checker_name, basepath, basename, top_left, bottom_right, numpieces, start_date):
        """
        Build oneself.
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
                        print time.asctime(), "Received: ", \
                            len(ret_photos), " Inserted: ", numinserted
                        if len(ret_photos) == 0 or numinserted == 0:
                            print "Marking ", target, " dead for ", \
                                self.basename
                            self.tgt_list[index]['dead'] = True

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
    return check


if __name__ == "__main__":

    colls = list()

    for yr in [2011, 2012, 2013, 2014]:
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
                                   10, dt.datetime(yr, 1, 1)))

    for i in colls:
        i.start()
