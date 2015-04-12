# coding: utf-8
from __future__ import (absolute_import, division, print_function, unicode_literals)

import logging
from borough import DataCollector as DC
import datetime as dt

if __name__ == "__main__":
    global LOGGER
    logging.basicConfig(filename='./logs/flickr_datacollect.log', level=logging.INFO)
    LOGGER = logging.getLogger()

    colls = list()

    # for yr in [2011, 2012, 2013, 2014]:
    for yr in [2006, 2007, 2008, 2009, 2010]:
        yrs = str(yr)
        colls.append(DC.DataCollector(logname=__name__, checker_name="./data/checkers/hampton_roads"+yrs+".pickle",
                                      basepath="./data/hroads"+yrs, basename="hr_data"+yrs,
                                      top_left=[37.91, -77.81], bottom_right=[35.8, -75.40],
                                      numpieces=10, start_date=dt.datetime(yr, 1, 1)))
        # colls.append(DataCollector("../data/checkers/sf_checker"+yrs+".pickle",
        #                            "./data/sfo"+yrs, "sfo_tgts_test"+yrs,
        #                            [38, -123], [37, -120],
        #                            5, dt.datetime(yr, 1, 1)))
        # colls.append(DataCollector("../data/checkers/nyc"+yrs+".pickle",
        #                            "./data/nyc/"+yrs, "nyc"+yrs,
        #                            [42, -75], [40, -72],
        #                            10, dt.datetime(yr, 1, 1)))
        # colls.append(DataCollector("../data/checkers/bigdc"+yrs+".pickle",
        #                            "./data/wdc/"+yrs, "wdc"+yrs,
        #                            [40, -78], [37.5, -76],
        #                            8, dt.datetime(yr, 1, 1)))
        # colls.append(DataCollector("./data/checkers/stisle"+yrs+".pickle",
        #                            "./data/stislestisle"+yrs, "stl"+yrs,
        #                            [40.6, -74.3], [40.4, -73.9],
        #                            3, dt.datetime(yr, 1, 1)))

    for i in colls:
        i.start()
