import csv
import sys

typename = str(sys.argv[1])

gsl_path = "../GSL_isol/"
final_dataset_file = '../' + typename + '_dataset.csv'
open(final_dataset_file,'w').close()

with open(final_dataset_file, "a") as datf:
    csvwriter = csv.writer(datf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    readname = '../' + typename + '_unique.csv'
    with open(readname,'r') as csvf:
        csvreader = csv.reader(csvf, delimiter=",")
        headers = next(csvreader, None)
        print(headers)

        for row in csvreader:
            if len(row) == 0: continue
            word_id = 0
            for video_path in row:
                if video_path == '':
                    word_id += 1
                    continue
                pathlist = [video_path,str(word_id)]
                csvwriter.writerow(pathlist)
                word_id += 1