#! /usr/bin/python

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox()
driver.get('https://mail.google.com/')

emailBox = driver.find_element_by_id('Email')
emailBox.clear()
emailBox.send_keys('desmith@mymail.mines.edu')

driver.find_element_by_id('next').click()

minesUsrNm = driver.find_element_by_id('username')
minesUsrNm.clear()
minesUsrNm.send_keys('desmith')

minesPasswrd = driver.find_element_by_id('password')
minesPasswrd.clear()
minesPasswrd.send_keys('AbAbId44')

driver.find_element_by_tag_name('Button').click()
