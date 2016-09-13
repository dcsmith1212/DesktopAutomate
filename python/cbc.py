#! /usr/bin/python

import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import sys

whichSport = sys.argv[1:]
sportId = ' '.join(sys.argv[1:])

sports 	 = {'archery':'AR',
			'artistic gymnastics':'GA',
			'beach volleyball':'BV',
			'boxing':'BX' }

driver = webdriver.Firefox()
driver.get('http://olympics.cbc.ca/online-listing/')

driver.find_element_by_xpath('//select[@class="or-select"]/option[@value="'+sports[sportId]+'"]').click()

driver.find_element_by_link_text('All').click()

bashVpnOpen = 'echo ginger44 | sudo -S nmcli con up id "PIA - CA Toronto"'
os.system(bashVpnOpen)
print 'Connecting to VPN:  PIA - CA Toronto...'
