# coding: utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)

import logging
from borough import DataCollector as DataColl
import datetime as dt

if __name__ == "__main__":
    global LOGGER
    logging.basicConfig(filename='./logs/flickr_datacollect.log', level=logging.INFO)
    LOGGER = logging.getLogger()

    colls = list()

    # for yr in [2011, 2012, 2013, 2014]:
    for yr in [2006, 2007, 2008, 2009, 2010]:
        yrs = str(yr)
        colls.append(DataColl.DataCollector(logname=__name__, checker_name="./data/checkers/hampton_roads"+yrs+".pickle",
                                            basepath="./data/hroads"+yrs, basename="hr_data"+yrs,
                                            top_left=[37.91, -77.81], bottom_right=[35.8, -75.40],
                                            numpieces=10, start_date=dt.datetime(yr, 1, 1)))

        colls.append(DataColl.DataCollector(logname=__name__, checker_name="./data/checkers/sf_checker"+yrs+".pickle",
                                            basepath="./data/sfo"+yrs, basename="sfo_tgts_test"+yrs,
                                            top_left=[38, -123], bottom_right=[37, -120],
                                            numpieces=5, start_date=dt.datetime(yr, 1, 1)))
        colls.append(DataColl.DataCollector(logname=__name__, checker_name="./data/checkers/nyc"+yrs+".pickle",
                                            basepath="./data/nyc/"+yrs, basename="nyc"+yrs,
                                            top_left=[42, -75], bottom_right=[40, -72],
                                            numpieces=10, start_date=dt.datetime(yr, 1, 1)))
        colls.append(DataColl.DataCollector(logname=__name__, checker_name="./data/checkers/bigdc"+yrs+".pickle",
                                            basepath="./data/wdc/"+yrs, basename="wdc"+yrs,
                                            top_left=[40, -78], bottom_right=[37.5, -76],
                                            numpieces=8, start_date=dt.datetime(yr, 1, 1)))
        colls.append(DataColl.DataCollector(logname=__name__, checker_name="./data/checkers/stisle"+yrs+".pickle",
                                            basepath="./data/stislestisle"+yrs, basename="stl"+yrs,
                                            top_left=[40.6, -74.3], bottom_right=[40.4, -73.9],
                                            numpieces=3, start_date=dt.datetime(yr, 1, 1)))

    for i in colls:
        i.start()
