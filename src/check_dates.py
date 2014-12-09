import json

infile = 'nyc_json_fll.json'
infile = open(infile, 'r')


maxdate = 0
for line in infile:
    try:
        line = json.loads(line)
        if int(line['dateupload']) > maxdate:
            maxdate = int(line['dateupload'])
    except ValueError:
        pass

print maxdate
