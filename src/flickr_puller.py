


import json
import requests
import time

API_KEY = "53de8b67e106afc4efbf8790d421dae8" 
NY_LAT = 40.69
NY_LON = 74.00
DC_LAT = 38.90
DC_LON = -77.00

def get_payload(city_lat, city_lon, c_page):
    payload = {"method": "flickr.photos.search",
            "accuracy": 8,
            "lat": city_lat,
            "lon": city_lon,
            "radius": 32,
            "extras": "date_upload,date_taken,owner_name,geo,tags",
            "per_page": 500,
            "format": "json",
            "nojsoncallback": 1,
            "page": c_page,
            "api_key": API_KEY }
    r = requests.get("https://api.flickr.com/services/rest/", params=payload)
    photoList = json.loads(r.text)
    return photoList["photos"]["photo"]

if __name__ == "__main__":
    nyc_file = file("nyc_json.json", 'w')
    wdc_file = file("wdc_json.json", 'w')

    for i in range(0, 100):
        print "Iteration: " + str(i)
        nyc_photos = get_payload(NY_LAT, NY_LON, i)
        for photo in nyc_photos:
            nyc_file.write(json.dumps(photo) + "\n")
        wdc_photos = get_payload(DC_LAT, DC_LON, i)
        for photo in wdc_photos:
            wdc_file.write(json.dumps(photo) + "\n")

    nyc_file.close()
    wdc_file.close()
