#! /usr/bin/python

import sys, requests, bs4, webbrowser
from abc import ABCMeta, abstractmethod

class Website(object):
	'Contains all components necessary to access a variety of websites'
	
	#__metaclass__ = ABCMeta

	baseSearchURL = ''
	delim = []
	defaultNum = 0
	selectString = ''
	baseLinkURL = ''
	linkContainer = ''
	
	def __init__(self, rawInput):
		# If first CL argument is a digit, open that many tabs
		# Otherwise use default value
		if len(sys.argv) > 1:
			if sys.argv[1].isdigit():
				self.num = int(sys.argv[1])
				self.search = self.delim.join(sys.argv[2:])
			else:
				self.num = self.defaultNum
				self.search = self.delim.join(sys.argv[1:])

	def request_page(self):
		# Access web page based on default URL and phrase searched
		# Syntax for custom search portion will vary per website 
		# Usually search are separated by some delimiter
		self.res = requests.get(self.baseSearchURL + self.search)
		self.res.raise_for_status()

	def parse_html(self):
		# Grab html from webpage and find contents of specific tag,
		# given by the attribute (and class?) type in the html code
		# [Try removing 'lxml' if fails on different system]
		self.html = bs4.BeautifulSoup(self.res.text, 'lxml')
		self.linkElems = self.html.select(self.selectString)

	def open_link(self):
		# Choose either the specified number of links for opening,
		# or the max number of links on the page if that's too big:
		print "Searching..."
		self.numOpen = min(self.num, len(self.linkElems))
		for i in range(self.numOpen):
			webbrowser.open_new_tab(self.baseLinkURL + 
						self.linkElems[i].get(self.linkContainer))

	def search_web(self):
		self.request_page()
		self.parse_html()
		self.open_link()

	# Might need this to use ABCMeta
	#@abstractmethod
	#def site_name(self):
	#	pass

#----------------------------------------------------------------------

class Google(Website):

	baseSearchURL = 'http://google.com/search?q='
	defaultNum = 1
	delim = '+'
	selectString = '.r a'
	baseLinkURL = 'http://google.com'
	linkContainer = 'href'
	
	def site_name(self):
		return 'google' 


class Stack(Google):

	baseSearchURL = 'http://google.com/search?q=insite:stackoverflow.com'

	def site_name(self):
		return 'stackoverflow'


class Wiki(Google):

	baseSearchURL = 'http://google.com/search?q=insite:wikipedia.org'

	def site_name(self):
		return 'wikipedia'

	

