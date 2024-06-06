import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service 
import requests
import time

# Base URL of your website
base_url = "https://kvk.icar.gov.in/"
base_addr = "p_prac.aspx"  # Replace with the actual URL

# Function to download a single PDF
def download_pdf(pdf_url, download_dir):
    file_name = pdf_url.split("\\")[-1]
    file_path = os.path.join(download_dir, file_name)


    with requests.get(pdf_url, stream=True, verify=False) as r:
        try :
            r.raise_for_status()  # Check for HTTP errors
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except:
            print("error")


# Create a download directory if it doesn't exist 
download_dir = "kvk_pop" 
os.makedirs(download_dir, exist_ok=True) 

WINDOW_SIZE = "1920,1080"
options = webdriver.ChromeOptions()
options.add_argument("user-data-dir=./new_chrome_profile")  # New profile directory
options.add_argument("--headless")
options.add_argument("--window-size=%s" % WINDOW_SIZE)
options.add_argument('--no-sandbox')
# Create a Service object
service = Service("/data2/home/sahilkamble/chromedriver") 

# Instantiate the driver
driver = webdriver.Chrome(service=service, options=options) 
driver.get(base_url+base_addr)

# Find the elements
state_select = Select(driver.find_element(By.ID, 'ContentPlaceHolder1_ddlState'))
s = 22 # Skip '--Select--'
sl = len(state_select.options)
# Iterate through states
while s < sl:
    state_select = Select(driver.find_element(By.ID, 'ContentPlaceHolder1_ddlState'))
    state_option = state_select.options[s]
    st = state_option.text
    os.makedirs(os.path.join(download_dir, st), exist_ok=True)
    print(st)
    state_option.click()
    # Wait for districts to load 
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "ContentPlaceHolder1_ddlDistrict"))
    )
    district_select = Select(driver.find_element(By.ID, 'ContentPlaceHolder1_ddlDistrict'))
    # Iterate through districts
    d = 1
    dl = len(district_select.options)
    while d < dl:
        district_select = Select(driver.find_element(By.ID, 'ContentPlaceHolder1_ddlDistrict'))
        district_option = district_select.options[d]
        dt = district_option.text
        os.makedirs(os.path.join(download_dir, st, dt), exist_ok=True) 
        print(dt)
        district_option.click()
        # Wait for KVKs to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "ContentPlaceHolder1_ddlKvk"))
        )
        kvk_select = Select(driver.find_element(By.ID, 'ContentPlaceHolder1_ddlKvk'))
        kl = len(kvk_select.options)
        k = 1
        # Iterate through KVKs
        while k < kl :
            kvk_select = Select(driver.find_element(By.ID, 'ContentPlaceHolder1_ddlKvk'))
            kvk_option = kvk_select.options[k]
            kt = kvk_option.text
            os.makedirs(os.path.join(download_dir, st, dt, kt), exist_ok=True)
            print(kt)
            kvk_option.click()
            submit_button = driver.find_element(By.ID, 'ContentPlaceHolder1_btnSubmit')
            submit_button.click()

            # Wait for results to load & extract PDF links
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "tblSample"))
            )

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            pdf_links = [link.get('href') for link in soup.find_all('a') 
             if link.get('href') and (link.get('href').endswith('.pdf') or  link.get('href').endswith('.PDF')) and link.get('href').startswith('API')] # Modified line 
            
            print(pdf_links)

            # Download PDFs
            for link in pdf_links:
                download_pdf(base_url + link, os.path.join(download_dir, st, dt, kt))  # Adjust URL creation if needed
                # time.sleep(1)  # Small delay to not overwhelm the server
            k+=1
        d+=1
    s+=1

driver.quit()
