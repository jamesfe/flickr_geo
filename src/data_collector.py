"""
A program to pull data over a set of coordinates  Uses flickr_puller
"""

import time
import os
import json
import pickle
from flickr_puller import connect, get_payload, import_line
import threading
import datetime as dt


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

    def run(self):
        """
        run the collector
        :return:
        """
        print "Starting.", self.checker
        connect()
        checker = get_checker(self.checker)
        self.pull_data(checker['curr_page'], checker['sec_time'],
                       self.stime, self.tgt_list)

    def pull_data(self, minpage, minday, start_time, targets):
        """
        generic function to grab more data
        :return:
        """
        totcount = 0
        for curr_eval in range(minpage, 5):  # number of pages left = 5
            base_time = start_time
            print self.basename, ": Page Pull: " + str(curr_eval)
            if curr_eval == minpage:
                md = minday
            else:
                md = 0

            for sec_time in range(md, 365):
                checker = file(self.checker, 'w')
                pickle.dump(dict({"sec_time": sec_time,
                                  "curr_page": curr_eval}), checker)
                checker.close()

                yearcount = 0
                delta_time = base_time + (sec_time * 86400)

                for target in targets:
                    ret_photos = get_payload(target['lat'], target['lon'],
                                             curr_eval, delta_time)
                    for photo in ret_photos:
                        target['file'].write(json.dumps(photo) + "\n")
                        r = import_line(photo)
                        if r is not False:
                            yearcount += 1
                            totcount += 1
                print self.basename, ": ", time.asctime(), "Day of Year: ", sec_time, " Photos: ", \
                    yearcount, totcount

        for target in targets:
            target['file'].close()


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
            r = dict({'lat': newlat, 'lon': newlon, 'fname': fname})
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
    # connect()
    #
    # # tgt_set = [dict({'coords': [[38, -123], [37, -120]],
    # #                  'basename': 'sfo_tgts'})]
    # # tgt_set = [dict({'coords': [[40.6, -74.3], [40.4, -73.9]],
    # #                  'basename': 'staten_isle'}),

    # ## bronx_checker.pickle
    # # tgt_set = [dict({'coords': [[41, -74], [40.5, -73.6]],
    # #                 'basename': 'bronx'})]
    #
    # # global CURR_CHECKER
    # # CURR_CHECKER = "./dctgts_checker.pickle"
    # # tgt_set = [dict({'coords': [[40, -78], [37, -75]], 'basename': 'dc_tgts_2010'}),
    # #            dict({'coords': [[42, -75], [39, -73]], 'basename': 'ny_tgts_2010'})]
    #
    # # global CURR_CHECKER
    # # pieces = 2
    # # CURR_CHECKER = "./stisle_2010_checker.pickle"
    # # tgt_set = [dict({'coords': [[40.6, -74.3], [40.4, -73.9]],
    # #                  'basename': 'staten_isle'})]
    #

    dc = DataCollector("./sf_checker_2011.pickle", "sfo", "sfo_tgts_test",
                       [38, -123], [37, -120],
                       5, dt.datetime(2011, 1, 1))

    hr = DataCollector("./hampton_roads.pickle", "hroads", "hr_data",
                       [37.91, -77.81], [35.8, -75.40],
                       10, dt.datetime(2010, 1, 1))

    dc.start()
    hr.start()

