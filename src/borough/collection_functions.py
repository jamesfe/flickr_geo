# coding: utf-8

import json
import os
import pickle
import requests
import time


def get_payload(city_lat=None, city_lon=None, c_page=None, api_key=None, min_date=None):
    """
    Get a number of photos within 32km of a point.
    :param city_lat: latitude in DD
    :param city_lon: longitude in DD
    :param c_page:  which page do you want? 1-n
    :return:
    """
    start = time.time()
    threshold = 25  # number of seconds we want a call to take
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
               "api_key": api_key}
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


def get_checker(fname, logger):
    """
    get where we last left off
    :param fname:
    :return:
    """
    if not os.path.isfile(fname):
        check = dict({u"sec_time": 1, u"curr_page": 1})
    else:
        with open(fname, 'rb') as infile:
            check = pickle.load(infile)

    logger.info(check)
    return check


