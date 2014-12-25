import unittest
import flickr_puller


class TestFlickerPuller(unittest.TestCase):
    def setUp(self):
        testfile = file("./tag_data.txt", 'r')
        self.testdata = testfile.read()

    def test_clean_tags(self):
        stopwords = flickr_puller.get_stopwords()

        nofinds = flickr_puller.get_findwords()

        cleantags = flickr_puller.clean_tags(self.testdata, stopwords, nofinds)
        for tag in set(cleantags.split(" ")):
            for nf in nofinds:
                self.assertEquals(tag.find(nf), -1)