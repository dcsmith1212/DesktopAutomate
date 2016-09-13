from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Firefox()
driver.get('https://mail.google.com/')

emailBox = driver.find_element_by_id('Email')
emailBox.clear()
emailBox.send_keys('dcsmith121244@gmail.com')

driver.find_element_by_id('next').click()

emailBox = driver.find_element_by_id('Passwd')
emailBox.clear()
emailBox.send_keys('ginger44')

driver.find_element_by_id('signIn').click()
