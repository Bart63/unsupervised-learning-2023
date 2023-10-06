import re
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, 'winnie_the_pooh.txt')

def read_n_characters_with_regex(file_path, n, start_line, regex_pattern):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_line = start_line
        result = ""

        while current_line < len(lines):
            line = lines[current_line]
            matches = re.findall(regex_pattern, line)
            for match in matches:
                result += match
                if len(result) >= n:
                    return result[:n]
            current_line += 1

    return result


def extract(nb_chars=50, start_line=190, regex_pattern=r'[a-zA-Z0-9]'):
    return read_n_characters_with_regex(FILE_PATH, nb_chars, start_line, regex_pattern)
