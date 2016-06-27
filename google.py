#! /usr/bin/python

from website import Google
import sys

siteObj = Google(sys.argv[1:])
siteObj.search_web()
