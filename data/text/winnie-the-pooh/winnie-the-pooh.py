import requests
import os


response = requests.get('http://www.gutenberg.org/cache/epub/1112/pg1112.txt')

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'winnie-the-pooh.txt')

with open(file_path, 'w') as f:
    f.write(response.text)
