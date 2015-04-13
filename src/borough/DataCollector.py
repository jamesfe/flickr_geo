# coding: utf-8
from __future__ import (division, unicode_literals, print_function)

"""
A program to pull data over a set of coordinates  Uses flickr_puller
"""

import time
import os
import json
import logging
import pickle
import threading
import psycopg2
from .collection_functions import (build_targets, get_checker, get_payload)

with open("api_key.txt", 'r') as apikey:
    API_KEY = apikey.read().strip()


class DataCollector(threading.Thread):
    """
    Derived from Thread so we ran run lots of these guys, this class defines a single collections task.
    """

    def __init__(self, logname=None, checker_name=None, basepath=None, basename=None, top_left=None,
                 bottom_right=None, numpieces=None, start_date=None):
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
        self.logger = logging.getLogger(logname)
        self.checker = checker_name
        self.basename = basename
        self.tgt_list = build_targets([top_left, bottom_right],
                                      numpieces, basename)
        self.stime = int(time.mktime(start_date.timetuple()))
        self.basepath = basepath

        self.numpieces = numpieces

        if not os.path.isdir(basepath):
            os.mkdir(basepath)
            self.logger.info("Made directory: " + basepath)

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
        self.logger.info("Starting " + self.checker)
        self.pull_data()

    def pull_data(self):
        """
        generic function to grab more data
        :return:
        """
        start_checker = get_checker(self.checker, self.logger)
        md = start_checker['sec_time']

        for sec_time in range(md, 365):
            self.logger.info(self.basename + " seconds: " + str(sec_time))
            minpage = 0
            if sec_time != start_checker['sec_time']:
                minpage = start_checker['curr_page']

            for curr_eval in range(minpage, 5):  # number of pages left = 5

                checker = open(self.checker, 'wb')
                pickle.dump(dict({"sec_time": sec_time,
                                  "curr_page": curr_eval}), checker)
                checker.close()
                delta_time = self.stime + (sec_time * 86400)

                for index, target in enumerate(self.tgt_list):
                    if target['dead'] is not True:
                        ret_photos = get_payload(target['lat'], target['lon'],
                                                 curr_eval, API_KEY, delta_time)
                        numinserted = 0
                        for photo in ret_photos:
                            target['file'].write(json.dumps(photo) + "\n")
                            r = self.import_line(photo)
                            if r is not False:
                                numinserted += 1
                        if len(ret_photos) == 0 or numinserted == 0:
                            self.tgt_list[index]['dead'] = True
                        elif numinserted > 0:
                            self.logger.info((self.basename + " received: ",
                                              str(len(ret_photos)) + " Inserted: " + str(numinserted)))

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
            except psycopg2.Error as err:
                self.logger.error("ERROR: {0} {1}".format(
                    err.diag.severity, err.pgerror))
        else:
            return False
        return line['id']

