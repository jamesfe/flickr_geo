Flickr Borough Analyzer
=======================

Introduction
------------

This is a tool that scratches one specific itch:
"If a neighborhood had to be in New York City, which borough would it fit in with the most?"

A burning question in all of our hearts, I know.

Method
------

We use data_collector.py to collect the target city as well as New York City.  Then, we train an SVM to know what photo metadata in each borough looks like: after that, we can throw metadata from other cities and their neighborhoods at it to see where they show up.

flickr_puller.py (a misnomer) is the analytical engine.  After we've run the data collector for some time, we have a Postgres database full of data, so we train an SVM with that data.  

There are a few helper functions that we can use to tune the SVM, specifically, find_overlapping_tags(trng_file, over_file), because it will tell us tags that aren't unique to a borough.  (This is risky business, fooling with the data.)  

We can add words we never want to appear to findwords.txt, and we can add words that must be erased in their entirety to stopwords.txt.  (For example, we see instagramapp:userid=39284987 a lot, which is meaningless to the SVM, so I add instagramapp to findwords.txt; "manhattan" in its entirety is bad, so I add that to stopwords.txt).

We have some spaghetti code to run to get everything to work:

    # Pre-compute where each picture is (which borough, basd on lat/lng)
    perform_geo_class_nyc()
    # Pick a file to send the output to
    trn_file = "./pickles/out_7000.pickle"
    # Find some training data
    gather_nyc_data([15000]*5, 'nyc_class', trn_file)
    print "Finding Overlaps: "
    # Find which tags overlap a lot
    find_overlapping_tags(trn_file, "overlaps.txt")
    print "Overlaps complete."
    # Store predictions in database
    train_to_database(-1, "total_redo", 7001, trn_file)
    hashlen = 5
    ofile = file("./webapp/data/dc_outfile_geohash"+str(hashlen)+".json", 'w')
    dc_lbox = dict({"lats": [40, 37.5], "lons": [-78, -76]})
    # Calculate out a visualization
    rvals = geohash_to_polygons(7001, 1000000, dc_lbox, hashlen)
    ofile.write("var inData =")
    ofile.write(json.dumps(rvals))
    ofile.write(";\n")
    ofile.close()
    print "done"
    # Finished
   
After all these steps have been run, we can check out webapp/index.html (which is also a misnomer; you can just see it locally.)

Questions?
==========
Twitter: @jimmysthoughts

Or make an issue here.