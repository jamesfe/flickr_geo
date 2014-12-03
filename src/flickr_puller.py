"""
A quick script to pull some data from flickr surrounding given points.
"""
import json
import requests
import time
import datetime
import fiona
from shapely.geometry import Point, shape

API_KEY = open("api_key.txt", 'r').read().strip()
NY_LAT = 40.69
NY_LON = -74.00
DC_LAT = 38.90
DC_LON = -77.00


def get_payload(city_lat, city_lon, c_page):
    """
    Get a number of photos within 32km of a point.
    :param city_lat: latitude in DD
    :param city_lon: longitude in DD
    :param c_page:  which page do you want? 1-n
    :return:
    """
    payload = {"method": "flickr.photos.search",
               "accuracy": 8,
               "lat": city_lat,
               "lon": city_lon,
               "radius": 32,
               "extras": "date_upload,date_taken,owner_name,geo,tags",
               "per_page": 250,
               "min_upload_date": "2014-01-01",
               "format": "json",
               "nojsoncallback": 1,
               "page": c_page,
               "api_key": API_KEY}
    flickr_req = requests.get("https://api.flickr.com/services/rest/",
                              params=payload)
    photo_list = json.loads(flickr_req.text)
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

    in_reader = open(in_file, 'r')
    tag_returns = list()
    count = 0
    totcount = 0
    for line in in_reader:
        try:
            jsline = json.loads(line)
            if jsline['owner'] in valid_authors:
                lat = jsline['latitude']
                lon = jsline['longitude']
                # It pains me to do the below two lines, but sometimes flickr
                # sends back geos that match NYC but the original data is bad
                # I think it has to do with their indexing being diffrent from
                # actual values?  But anyways, we need the data and the tags
                # make sense for NYC.
                if lon > 0:
                    lon *= -1
                tgt_point = Point(lon, lat)  # silly shapely - Point(x,y,z)
                new_val = list()
                # if not tgt_point.intersects(not_nyc_shp):
                for index, geo in enumerate(geoms):
                    if tgt_point.intersects(geo[0]):
                        new_val.append(index)
                        break
                if len(new_val) == 0:
                    count += 1

                split_tags = jsline['tags'].split(" ")
                if len(new_val) > 0 and len(split_tags) > 1:
                    new_val.append(split_tags)
                    tag_returns.append(new_val)
        except ValueError:
            pass
    print count
    print len(tag_returns)
    print tag_returns
    tot_time = time.time() - starttime
    print tot_time
    return tag_returns


def pull_data():
    """
    generic function to grab more data
    :return:
    """

    nyc_file = open("nyc_json_fll.json", 'a')
    wdc_file = open("wdc_json_fll.json", 'a')

    for i in range(1, 10000):
        print "Iteration: " + str(i)
        nyc_photos = get_payload(NY_LAT, NY_LON, i)
        print "NYC Photos: " + str(len(nyc_photos))
        for photo in nyc_photos:
            nyc_file.write(json.dumps(photo) + "\n")
        wdc_photos = get_payload(DC_LAT, DC_LON, i)
        print "WDC Photos: " + str(len(wdc_photos))
        for photo in wdc_photos:
            wdc_file.write(json.dumps(photo) + "\n")

    nyc_file.close()
    wdc_file.close()

if __name__ == "__main__":
    # pic_count = 0
    # tot_count = 0
    tgt_nyc_file = "nyc_json.json"
    nyc_authors = multi_day_authors(tgt_nyc_file, 5)
    print len(nyc_authors)
    ret_tags = return_borough_tags(nyc_authors, tgt_nyc_file)
    # for k in open(tgt_nyc_file, 'r'):
    #     r = json.loads(k)
    #     tot_count += 1
    #     if r['owner'] in nyc_authors:
    #         pic_count += 1
    # print pic_count, tot_count
    # pull_data()
    # print nyc_authors

    print "done"


    # {"ispublic": 1, "place_id": "5ebRtgVTUro3guGorw", "geo_is_public": 1,
    # "owner": "72098626@N00", "id": "15604638529", "title": "Skateboard kid #8",
    # "woeid": "20070197", "geo_is_friend": 0, "geo_is_contact": 0, "datetaken":
    # "2014-09-28 16:02:01", "isfriend": 0, "secret": "abf65e17a8", "ownername":
    # "Ed Yourdon", "latitude": 40.731371, "accuracy": "16", "isfamily": 0,
    # "tags": "newyork greenwichvillage streetsofnewyork streetsofny everyblock",
    # "farm": 8, "geo_is_family": 0, "dateupload": "1416009675",
    # "datetakengranularity": "0", "longitude": 74.005447, "server": "7477", "context": 0}
