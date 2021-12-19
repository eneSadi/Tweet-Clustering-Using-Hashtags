import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--since', type=str, required=True)
parser.add_argument('--until', type=str, required=True)
parser.add_argument('--keywords', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

def scrape_tw(since, until, keywords, output_dir):
  '''
    - since (str)      : starting date for scraping. Exp: 2020-01-01 15:30:00
    - until (str)      : ending date for scraping.   Exp: 2021-01-01 16:00:00
    - keywords (list)  : which words do you want to scrape on twitter.
    - output_dir (str) : the path where the csv file will be saved.
  '''

  keys = keywords[0]
  if len(keywords) > 1:
    for keyword in keywords[1:]:
      keys += " OR " + keyword
  print("Keys: " + keys)

  first_until = until
  uncut_output = output_dir + '/' + keywords[0] + '_' + since.split()[0] + '_' + since.split()[1] + '_' + first_until.split()[0] + '_' + first_until.split()[1] + '.csv'

  keep = 0
  while keep == 0:
    output = output_dir + '/' + keywords[0] + '_' + since.split()[0] + '_' + since.split()[1] + '_' + until.split()[0] + '_' + until.split()[1] + '.csv'
    call = 'twint -o "' + output + '" --csv --tabs -s "' + keys + '" --since "' + since + '" --until "' + until + '"'
    a = os.system(call)
    print(a, call)

    df = pd.read_csv(output, sep = '\t')
    ctrl = df.iloc[-1]['date'] + ' ' + df.iloc[-1]['time'] 
    print(ctrl)
    if ctrl.split()[0] == since.split()[0] and ctrl.split()[1].split(':')[0] == since.split()[1].split(':')[0] and ctrl.split()[1].split(':')[1] == since.split()[1].split(':')[1]:
      if output == uncut_output:
        print("File saved here: " + uncut_output)
      else:
        print("File saved here: " + output)
      keep = 1
    else:
      new_name = output_dir + '/' + keywords[0] + '_' + ctrl.split()[0] + '_' + ctrl.split()[1] + '_' + until.split()[0] + '_' + until.split()[1] + '.csv'
      os.rename(output, new_name)
      print("File saved here: " + new_name)
      until = ctrl


if __name__ == "__main__":

    since = args.since
    print(since)
    until = args.until
    print(until)
    keywords = args.keywords.split()
    print(keywords)
    output_dir = args.output_dir
    print(output_dir)

    scrape_tw(since, until, keywords, output_dir)
