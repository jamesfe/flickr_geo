# coding: utf-8
"""
quick loader for json files -> postgresql
"""
from __future__ import (absolute_import, division, unicode_literals, print_function)

import os
from .flickr_puller import file_to_database


if __name__ == "__main__":
    inpath = "./servicedown/"
    for k in os.listdir(inpath):
        file_to_database(os.path.join(inpath, k))