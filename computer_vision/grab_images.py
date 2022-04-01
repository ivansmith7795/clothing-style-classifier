import pandas as pd
import numpy as np
import urllib.request as urllib2
from bs4 import BeautifulSoup


def scrape_image(index, dress_page, style_class):
    #dress_page = dress_page.replace(" ","")

    page = None

    try:
        page = urllib2.urlopen(dress_page)
        
    except urllib2.HTTPError as e:
        print(str(index) + ' could not find: ' + dress_page + '. Failed with error code: ' + str(e.code))
    
    # Parse the page if returned data
    if page is not None:

        soup = BeautifulSoup(page, 'html.parser')

        images = []
        for img in soup.findAll('img', attrs={'class': 'FxZV-M'}):
            if "packshot" not in img.get('src'): 
                continue
            else:
                images.append(img.get('src'))
                break

        if len(images) > 0:
            urllib2.urlretrieve(images[0], "captured/" + str(index) + "_" + str(style_class) + ".jpg")

#Import into pandas
labeled_dataset = pd.read_csv('../datasets/dataset_dresses_labeled.csv')

for index, row in labeled_dataset.iterrows():
    page_url = row['link']
    style_class = row['Styles']
    index_id = row['Original order']
    scrape_image(index_id, page_url, style_class)
