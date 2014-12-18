"""
quick loader for json files -> postgresql
"""

import os
from flickr_puller import file_to_database

if __name__ == "__main__":
    inpath = "./servicedown/"
    for k in os.listdir(inpath):
        file_to_database(os.path.join(inpath, k))