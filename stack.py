#! /usr/bin/python

from website import Stack
import sys

siteObj = Stack(sys.argv[1:])
siteObj.search_web()
