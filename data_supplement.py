from bs4 import BeautifulSoup
import requests
import re
import urllib.request as urllib2
import os
import http.cookiejar as cookielib
import json
from image_utils import display_image, unpreprocess_image, load_image, image_dir


def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')


def get_image_samples(query):
    image_type="ActiOn"
    web_query= (query+' face').split()
    web_query='+'.join(web_query)
    url="https://www.google.co.in/search?q="+web_query+"&source=lnms&tbm=isch"
    print(url)
    #add the directory for your image here
    DIR="Pictures"
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
    }
    soup = get_soup(url,header)


    ActualImages=[]# contains the link for Large original images, type of  image
    for a in soup.find_all("div",{"class":"rg_meta"}):
        link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
        if len(ActualImages) < 10:
            ActualImages.append((link,Type))

    print ("there are total" , len(ActualImages),"images")

    if not os.path.exists(DIR):
                os.mkdir(DIR)
    DIR = os.path.join(DIR, web_query.split()[0])

    if not os.path.exists(DIR):
                os.mkdir(DIR)
    ###print images
    for i , (img , Type) in enumerate( ActualImages):
        try:
            req = urllib2.Request(img, headers={'User-Agent' : header})
            raw_img = urllib2.urlopen(req.full_url).read()
            cntr = len([i for i in os.listdir(DIR) if image_type in i]) + 1
            print(cntr)
            if len(Type)==0:
                filename = os.path.join(DIR , image_type + "_"+ str(cntr)+".jpg")
                f = open(filename, 'wb')
            else :
                filename = os.path.join(DIR , image_type + "_"+ str(cntr)+"."+Type)
                f = open(filename, 'wb')
            f.write(raw_img)
            f.close()
            target_path = os.path.join(image_dir, "_".join(query.split(' ')), 'supplement.jpg')
            print("TARGET_PATH: ", target_path)
            display_image(unpreprocess_image(load_image(filename)))
            for root, dir, files in os.walk(os.path.join(image_dir, "_".join(query.split(' ')))):
                for file in files:
                    display_image(unpreprocess_image(load_image(os.path.join(root, file))))
            ans = input("KEEP THIS IMAGE?: ")
            if ans == 'y':
                os.rename(filename, target_path)
                break
        except Exception as e:
            print("could not load : "+img)
            print(e)

sing = 0
mult = 0
single_folders = []
for root, dir, files in os.walk(image_dir):
    count = 0
    for file in files:
        if file.split('.')[-1] == 'jpg':
            count += 1
            # display_image(unpreprocess_image(load_image(os.path.join(root, file))))
    if count > 1:
        mult +=1
    else:
        sing +=1
        if len(root.split('/')) > 2:
            single_folders.append(' '.join(root.split('/')[-1].split('_')))
print("MULT: ", mult)
print("SING: ", sing)
print(single_folders[:10])
for name in single_folders:
    get_image_samples(name)
