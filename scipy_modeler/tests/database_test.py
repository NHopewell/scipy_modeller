"""
@author: Nicholas Hopewell
created on: May 28, 2019

Unit tests for database.py 
"""
import unittest
from pipeline.AALDatabase_ import Database
from random import choice

class DataBaseTest(unittest.TestCase):
    def test_connect(self):
        """tests truthyness of connection"""
        self.assertTrue()
        pass

    def test_badconn(self):
        """tests that bad connection info throws value error."""
