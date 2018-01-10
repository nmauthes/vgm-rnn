'''

Utility for scraping large numbers of MIDI files from VGMusic.com, for use during RNN training.

:authors Nicolas Mauthes

'''

import os
import argparse

import requests
from requests.compat import urljoin

from bs4 import BeautifulSoup


# Shortcuts to the URLs for various consoles' pages on VGMusic.com (e.g. 'nes' for Nintendo Entertainment System)
VGMUSIC_URLS = {'nes': 'https://www.vgmusic.com/music/console/nintendo/nes/',
                'snes': 'https://www.vgmusic.com/music/console/nintendo/snes/',
                'genesis': 'https://www.vgmusic.com/music/console/sega/genesis/'
                }


def get_midi_files(args):
    try:
        args.url = VGMUSIC_URLS[args.url]
    except KeyError:
        # You must really enjoy typing...
        pass

    source = requests.get(args.url).text  # TODO try/except
    soup = BeautifulSoup(source, 'lxml')

    links = soup.find_all('a', href=True)
    links = [l for l in links if l['href'].endswith('.mid')]

    if not os.path.exists(args.data_folder):
        os.mkdir(args.data_folder)

    print(f'Found {len(links)} MIDI files at {args.url}')

    errors = 0
    for i, link in enumerate(links[:args.max_files]):
        print(f'Downloading file {i + 1} of {len(links) if args.max_files >= len(links) else args.max_files}')

        try:
            resp = requests.get(urljoin(args.url, link['href']))
            open(os.path.join(args.data_folder, link['href']), 'wb').write(resp.content)
        except requests.exceptions.Timeout:
            errors += 1

    if errors:
        print('{errors} errors occurred')

parser = argparse.ArgumentParser(
    description='Utility for scraping MIDI files from VGMusic.com',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='See the README for for the questions'
)

parser.add_argument(
    'url',
    help='The URL to scrape MIDI files from. These should be links to a specific console\'s page on VGMusic.com.'
)
parser.add_argument(
    'data_folder',
    default='data',
    help='The name or path to the folder in which to download the MIDI files. Default is \'data\'.'
)

parser.add_argument(
    '--max_files',
    type=int,
    default=10000,
    help='The number of files to download from a given URL. Default is 10,000.'
)


if __name__ == '__main__':
    args = parser.parse_args()
    get_midi_files(args)