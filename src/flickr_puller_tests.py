# coding: utf-8
"""
The horrible python tests I've written for this.

Needs improvement.
"""
from __future__ import (absolute_import, division, unicode_literals, print_function)

import unittest
import flickr_puller


class TestFlickerPuller(unittest.TestCase):
    def setUp(self):
        testfile = open("./tag_data.txt", 'r')
        self.testdata = testfile.read()

    def test_clean_tags(self):
        stopwords = flickr_puller.get_stopwords()

        nofinds = flickr_puller.get_findwords()

        cleantags = flickr_puller.clean_tags(self.testdata, stopwords, nofinds)
        for tag in set(cleantags.split(" ")):
            for nf in nofinds:
                self.assertEqual(tag.find(nf), -1)