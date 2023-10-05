import requests
import os


URL = 'http://www.gutenberg.org/cache/epub/1112/pg1112.txt'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'winnie_the_pooh.txt')

def prepare():
    response = requests.get(URL)
    with open(FILE_PATH, 'w') as f:
        f.write(response.text)
    print('Winnie the pooh was downloaded!')
