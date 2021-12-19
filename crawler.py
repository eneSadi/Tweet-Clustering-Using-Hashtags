import os
import pandas as pd
import argparse

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
      keep = 1

      
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-odir", help="Directory for .csv files", type = str, default = "./data")
    parser.add_argument("--since", "-s", help="since date (type -> YYYY-MM-DD HH:MM:SS)", type=str)
    parser.add_argument("--until", "-u", help="until date (type -> YYYY-MM-DD HH:MM:SS)", type=str)
    parser.add_argument("--keywords", "-k", help="keywords to search, must be passed with dash (-) -> eth-ethereum", type=str)
    args = parser.parse_args()

    keys = args.keywords.split('-')

    print('Beginning of scraping.')
    scrape_tw(args.since, args.until, keys, args.output_dir)
    print('DONE!')
