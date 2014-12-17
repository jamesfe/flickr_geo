"""
quick loader for json files -> postgresql
"""

import os
from flickr_puller import file_to_database

if __name__ == "__main__":
    for k in os.listdir("./fix_megapull_2012/"):
        file_to_database("./fix_megapull_2012/" + k)