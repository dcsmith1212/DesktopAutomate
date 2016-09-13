#! /usr/bin/python

from website import Website
import sys

siteObj = Website(sys.argv[1:])
siteObj.baseSearchURL = 'https://mail.google.com/mail/u/0/#inbox'
