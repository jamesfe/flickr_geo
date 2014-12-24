"""
A program to pull data over a set of coordinates  Uses flickr_puller
"""

import time
import os
import json
import pickle
from flickr_puller import connect, get_payload, import_line


def build_targets(inpt, num, basename):
    lats = list([inpt[0][0], inpt[1][0]])
    lons = list([inpt[0][1], inpt[1][1]])
    maxlat = max(lats)
    minlat = min(lats)
    maxlon = max(lons)
    minlon = min(lons)
    latdiff = maxlat - minlat
    londiff = maxlon - minlon
    retvals = list()
    for latval in range(0, num):
        for lonval in range(0, num):
            newlat = minlat + (latdiff / num) * latval
            newlon = minlon + (londiff / num) * lonval
            fname = basename + "_" + str(latval) + "_" + str(lonval) + ".json"
            r = dict({'lat': newlat, 'lon': newlon, 'fname': fname})
            retvals.append(r)
    return retvals


def pull_data(minpage, minday, start_time, basepath, targets):
    """
    generic function to grab more data
    :return:
    """
    if not os.path.isdir(basepath):
        os.mkdir(basepath)
        print basepath

    for target in targets:
        tm = os.path.join(basepath,
                          str(time.time()).split(".")[0] + target['fname'])
        target['file'] = open(tm, 'a')

    totcount = 0
    for curr_eval in range(minpage, pieces):
        base_time = start_time
        print "Page Pull: " + str(curr_eval)
        if curr_eval == minpage:
            md = minday
        else:
            md = 0

        for sec_time in range(md, 365):
            checker = file(CURR_CHECKER, 'w')
            pickle.dump(dict({"sec_time": sec_time, "curr_page": curr_eval}), checker)
            checker.close()

            yearcount = 0
            dt = base_time + (sec_time * 86400)

            for target in targets:
                ret_photos = get_payload(target['lat'], target['lon'],
                                         curr_eval, dt)
                for photo in ret_photos:
                    target['file'].write(json.dumps(photo) + "\n")
                    r = import_line(photo)
                    if r is not False:
                        yearcount += 1
                        totcount += 1
            print time.asctime(), "Day of Year: ", sec_time, " Photos: ", \
                yearcount, totcount

    for target in targets:
        target['file'].close()


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
    connect()
    pieces = 2
    # tgt_set = [dict({'coords': [[38, -123], [37, -120]],
    #                  'basename': 'sfo_tgts'})]
    # tgt_set = [dict({'coords': [[40.6, -74.3], [40.4, -73.9]],
    #                  'basename': 'staten_isle'}),

    global CURR_CHECKER
    CURR_CHECKER = "./sf_checker.pickle"
    tgt_set = [dict({'coords': [[38, -123], [37, -120]],
                     'basename': 'sfo_tgts'})]

    # global CURR_CHECKER
    # CURR_CHECKER = "./hampton_roads.pickle"
    # tgt_set = [dict({'coords': [[37.91, -77.81], [35.8, -75.40]],
    #                  'basename': 'hampton_rds'})]

    ## bronx_checker.pickle
    # tgt_set = [dict({'coords': [[41, -74], [40.5, -73.6]],
    #                 'basename': 'bronx'})]

    # global CURR_CHECKER
    # CURR_CHECKER = "./dctgts_checker.pickle"
    # tgt_set = [dict({'coords': [[40, -78], [37, -75]], 'basename': 'dc_tgts_2010'}),
    #            dict({'coords': [[42, -75], [39, -73]], 'basename': 'ny_tgts_2010'})]

    global CURR_CHECKER
    CURR_CHECKER = "./stisle_2010_checker.pickle"
    tgt_set = [dict({'coords': [[40.6, -74.3], [40.4, -73.9]],
                     'basename': 'staten_isle'})]

    tgt_list = list()
    for i in tgt_set:
        tgt_list.extend(build_targets(i['coords'], pieces, i['basename']))
    # stime = 1325379661  # 2012
    # stime = 1357002061  # 2013
    stime = 1262307661  # 2010

    print tgt_set

    while True:
        try:
            checker = get_checker(CURR_CHECKER)
            pull_data(checker['curr_page'], checker['sec_time'],
                      stime, "./sfo/", tgt_list)
        except:
            print "Crashed!  Starting over."
            pass


            # TODO: Make this a class, instantiate the class,
            # which will fork off and use its own API Key.