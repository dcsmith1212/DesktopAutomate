#! /usr/bin/python

from website import Wiki
import sys

siteObj = Wiki(sys.argv[1:])
siteObj.search_web()


