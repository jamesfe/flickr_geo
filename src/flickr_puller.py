"""
A quick script to pull some data from flickr surrounding given points.
"""
import json
import requests
import time
import datetime

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
    # tgt_nyc_file = "nyc_json.json"
    # nyc_authors = multi_day_authors(tgt_nyc_file, 14)
    # for k in open(tgt_nyc_file, 'r'):
    #     r = json.loads(k)
    #     tot_count += 1
    #     if r['owner'] in nyc_authors:
    #         pic_count += 1
    # print pic_count, tot_count
    # pull_data()
    print "done"