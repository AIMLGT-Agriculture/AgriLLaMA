import requests
from bs4 import BeautifulSoup
import re
import os
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
downloads = 0
skipped_links = []
cookie = None

# we will just skip these, may be scrape them with selenium in future
login_req_list = ['https://epubs.icar.org.in/index.php/IJHF/issue/archive', 'https://epubs.icar.org.in/index.php/Agropedology/issue/archive',
                  'https://epubs.icar.org.in/index.php/IJVASR/issue/archive', 'https://epubs.icar.org.in/index.php/JIFSI/issue/archive',
                  'https://epubs.icar.org.in/index.php/TJRP/issue/archive', 'https://epubs.icar.org.in/index.php/IJAN/issue/archive', 
                  'https://epubs.icar.org.in/index.php/OIJR/issue/archive', 'https://epubs.icar.org.in/index.php/JSWC/issue/archive',
                  'https://epubs.icar.org.in/index.php/IJPP/issue/archive', 'https://epubs.icar.org.in/index.php/IPPJ/issue/archive',
                  'https://epubs.icar.org.in/index.php/jasa/issue/archive', 'https://epubs.icar.org.in/index.php/APPS/issue/archive',
                  'https://epubs.icar.org.in/index.php/VIB/issue/archive', 'https://epubs.icar.org.in/index.php/AERR/issue/archive',
                  'https://epubs.icar.org.in/index.php/JCMSD/issue/archive', 'https://epubs.icar.org.in/index.php/JAEM/issue/archive',
                  'https://epubs.icar.org.in/index.php/TR/issue/archive']


def download_pdf(pdf_url, retry_count=0):
    global downloads
    global cookie
    pattern = re.compile(r'/index\.php/([^/]+)')
    pdf_dir = re.search(pattern, pdf_url).group(1)
    pdf_name = pdf_url.split('/')[-1] + '.pdf'
    pdf_path = os.path.join(pdf_dir, pdf_name)
    downloads += 1
    if not os.path.exists(pdf_path):
        try:
            response = requests.get(pdf_url, verify=False, timeout=10, cookies=cookie)  # Set a timeout of 10 seconds
        except requests.exceptions.ReadTimeout:
            if(retry_count <= 10):
                print(f'Timeout occurred while fetching {pdf_url}, retrying...')
                return download_pdf(pdf_url, retry_count+1)  # Retry the function
            else:
                print(f'Timeout occurred while fetching {pdf_url}, stopped retrying')
                return
        except requests.exceptions.RequestException as e:
            print(f'An error occurred: {e}')
            return  # Exit the function on other request exceptions
        
        os.makedirs(pdf_dir, exist_ok=True)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded {pdf_name} to {pdf_path}, total  pdfs: {downloads}')
    else:
        print(f'File {pdf_name} already exists at {pdf_path}, skipping download, total pdfs: {downloads}')


def get_soup(link, retry_count=0):
    global skipped_links
    global cookie
    try:
        response = requests.get(link, verify=False, cookies=cookie)
    except :
        if(retry_count <= 10):
            print(f'Error occurred while fetching {link}, retrying...')
            return get_soup(link, retry_count=retry_count+1)  # Retry the function
        else:
            print(f'Error occurred while fetching {link}, stopped retrying')
            skipped_links.append(link)
            return
    return BeautifulSoup(response.text, 'html.parser')

def get_links(base_link, key) :
    soup = get_soup(base_link)
    if(soup) :
        return [a['href'] for a in soup.find_all('a', href=True) if key in a['href']]
    else:
        print(f'error while parsing {base_link}')


def download_pdfs(archive_url):
    pg_no = 1
    page_url = f'{archive_url}/{pg_no}'
    while(True):
        issue_links = get_links(page_url, 'issue/view')
        pg_no+=1
        page_url = f'{archive_url}/{pg_no}'
        if(not issue_links) :
            print(f'error on parsing {page_url} exiting')
            break
        for issue_link in issue_links:
            pdf_links = get_links(issue_link,'article/view')
            if(not pdf_links):
                    continue
            for pdf_link in pdf_links:
                download_links = get_links(pdf_link,'article/download' )
                if(not download_links):
                    continue
                for download_link in download_links:
                    download_pdf(download_link)


base_url = 'https://epubs.icar.org.in'
base_pg = requests.get(base_url, verify=False)

base_pg_soup = BeautifulSoup(requests.get(base_url, verify=False).text, 'html.parser')
journal_links = [a['href'] for a in base_pg_soup.find_all('a', href=True) if '/issue/current' in a['href']]
archive_links = [link.replace('/current', '/archive') for link in journal_links]
# print(archive_links[13:])
for archive_link in archive_links[:21]:
    if(archive_link not in login_req_list):
        # print(archive_link)
        download_pdfs(archive_link)

# https://epubs.icar.org.in/index.php/IJPP/issue/archive
print(f"downloads: {downloads} pdfs")
print(f"skipped: {skipped_links}")



# [{"domain":".epubs.icar.org.in","expirationDate":1700818693.736987,"hostOnly":false,"httpOnly":true,"name":"OJSSID","path":"/","sameSite":"lax","secure":false,"session":false,"storeId":"0","value":"irr4q3d5269skdv3u21dodbdvm"}]