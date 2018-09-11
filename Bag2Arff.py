#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
################################################################################
##              Laboratory of Computational Intelligence (LABIC)              ##
##             --------------------------------------------------             ##
##       Originally developed by: João Antunes  (joao8tunes@gmail.com)        ##
##       Laboratory: labic.icmc.usp.br    Personal: joaoantunes.esy.es        ##
##                                                                            ##
##   "Não há nada mais trabalhoso do que viver sem trabalhar". Seu Madruga    ##
################################################################################

import time
import datetime
import codecs
import logging
import os
import sys
import argparse
import math
import scipy.sparse


################################################################################
### FUNCTIONS                                                                ###
################################################################################

# Print iterations progress: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, estimation, prefix='   ', decimals=1, bar_length=100, final=False):
    columns = 32    #columns = os.popen('stty size', 'r').read().split()[1]    #Doesn't work with nohup.
    eta = str( datetime.timedelta(seconds=max(0, int( math.ceil(estimation) ))) )
    bar_length = int(columns)
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s %s%s |%s| %s' % (prefix, percents, '%', bar, eta))

    if final == True:    #iteration == total
        sys.stdout.write('\n')

    sys.stdout.flush()


#Format a value in seconds to "day, HH:mm:ss".
def format_time(seconds):
    return str( datetime.timedelta(seconds=max(0, int( math.ceil(seconds) ))) )


#Convert a string value to boolean:
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("invalid boolean value: " + "'" + v + "'")


#Verify if a value correspond to a natural number (it's an integer and bigger than 0):
def natural(v):
    try:
        v = int(v)

        if v > 0:
            return v
        else:
            raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")
    except ValueError:
        raise argparse.ArgumentTypeError("invalid natural number value: " + "'" + v + "'")


#Verify if a value correspond to a natural number (it's an integer and bigger than 0):
def percentage(v):
    try:
        v = float(v)

        if v >= 0 and v <= 1:
            return v
        else:
            raise argparse.ArgumentTypeError("invalid percentage number value: " + "'" + v + "'")
    except ValueError:
        raise argparse.ArgumentTypeError("invalid percentage number value: " + "'" + v + "'")


def get_index(v, l):
    return l.index(v) if v in l else None


################################################################################


################################################################################

#URL: https://github.com/joao8tunes/Bag2Arff

#Example usage: python3 Bag2Arff.py --token - --input out/Bag/txt/ --output out/Bag/arff/

#Defining script arguments:
parser = argparse.ArgumentParser(description="Doc-Attribute matrix to Arff file (Weka) converter\n==================================================")
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
optional.add_argument("--log", metavar='BOOL', type=str2bool, action="store", dest="log", nargs="?", const=True, default=False, required=False, help='display log during the process: y, [N]')
optional.add_argument("--print_features", metavar='BOOL', type=str2bool, action="store", dest="print_features", nargs="?", const=True, default=True, required=False, help='print features in Doc-Attribute header: [Y], n')
required.add_argument("--token", metavar='STR', type=str, action="store", dest="token", required=False, nargs="?", const=True, help='special "token" to split classes (e.g.: 1st-2nd)')
required.add_argument("--input", "-i", metavar='DIR_PATH', type=str, action="store", dest="input", required=True, nargs="?", const=True, help='input directory of Doc-Attribute files')
required.add_argument("--output", "-o", metavar='DIR_PATH', type=str, action="store", dest="output", required=True, nargs="?", const=True, help='output directory to save the Arff files')
args = parser.parse_args()    #Verifying arguments.

################################################################################


################################################################################

#Setup logging:
if args.log:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

total_start = time.time()

################################################################################


################################################################################
### INPUT (LOAD FILES LIST)                                                  ###
################################################################################

print("\nDoc-Attribute matrix to Arff file (Weka) converter\n==================================================\n\n\n")

if not os.path.exists(args.input):
    print("ERROR: input directory does not exists!\n\t!Directory: " + args.input)
    sys.exit()

print("> Loading input filepaths...\n\n\n")
files_list = []

#Loading all filepaths from root directory:
for file_item in os.listdir(args.input):
    files_list.append(args.input + file_item)

files_list.sort()
filepath_i = 0
total_num_examples = len(files_list)
eta = 0

if not os.path.exists(args.output):
    print("> Creating directory to convert...\n")
    os.makedirs(os.path.abspath(args.output), mode=0o755)

print("> Converting Doc-Attribute matrices:")
print("..................................................")
print_progress(filepath_i, total_num_examples, eta)
operation_start = time.time()

for filepath in files_list:
    start = time.time()
    file_name = filepath.split("/")[-1]
    file_item = codecs.open(filepath, "r", "utf-8")
    arff_file_first_second = codecs.open(args.output + file_name + '.arff', "w", "utf-8")

    if args.token is not None:
        arff_file_first = codecs.open(args.output + file_name.replace("_1st-2nd_", "_1st_") + '.arff', "w", "utf-8")
        arff_file_second = codecs.open(args.output + file_name.replace("_1st-2nd_", "_2nd_") + '.arff', "w", "utf-8")

    n_docs, n_feats = file_item.readline().split()
    n_docs = int(n_docs)
    n_feats = int(n_feats)
    matrix = scipy.sparse.lil_matrix((n_docs, n_feats))
    header = file_item.readline()
    arff_file_first_second.write("@relation " + file_name + ".arff\n\n")

    if args.token is not None:
        arff_file_first.write("@relation " + file_name.replace("_1st-2nd_", "_1st_") + ".arff\n\n")
        arff_file_second.write("@relation " + file_name.replace("_1st-2nd_", "_2nd_") + ".arff\n\n")

    if args.print_features is False:
        for d_i in range(1, n+1):
            arff_file_first_second.write("@attribute f" + str(d_i) + " numeric\n")

            if args.token is not None:
                arff_file_first.write("@attribute f" + str(d_i) + " numeric\n")
                arff_file_second.write("@attribute f" + str(d_i) + " numeric\n")
    else:
        data = header.split("\t")[:-1]

        for d in data:
            arff_file_first_second.write("@attribute " + d + " numeric\n")

            if args.token is not None:
                arff_file_first.write("@attribute " + d + " numeric\n")
                arff_file_second.write("@attribute " + d + " numeric\n")

    arff_file_first_second.write("@attribute class_atr {")

    if args.token is not None:
        arff_file_first.write("@attribute class_atr {")
        arff_file_second.write("@attribute class_atr {")

    features_first_second = []
    features_first = []
    features_second = []
    class_atr_first_second_i = []
    class_atr_first_i = []
    class_atr_second_i = []
    doc_i = 0

    for line in file_item:
        data = line.replace("\r\n", "").replace("\n", "").split("\t")
        class_atr_first_second = data.pop(-1)

        if args.token is not None:
            classes = class_atr_first_second.split(args.token)
            class_atr_first = "'" + classes[0] + "'"
            class_atr_second = "'" + classes[1] + "'"

        class_atr_first_second = "'" + class_atr_first_second + "'"

        for d_i, d in enumerate(data):
            matrix[doc_i, d_i] = d

        first_second_i = get_index(class_atr_first_second, features_first_second)

        if args.token is not None:
            first_i = get_index(class_atr_first, features_first)
            second_i = get_index(class_atr_second, features_second)

        if first_second_i is None:
            class_atr_first_second_i.append(len(features_first_second))
            features_first_second.append(class_atr_first_second)
        else:
            class_atr_first_second_i.append(first_second_i)

        if args.token is not None:
            if first_i is None:
                class_atr_first_i.append(len(features_first))
                features_first.append(class_atr_first)
            else:
                class_atr_first_i.append(first_i)

            if second_i is None:
                class_atr_second_i.append(len(features_second))
                features_second.append(class_atr_second)
            else:
                class_atr_second_i.append(second_i)

        doc_i += 1

    arff_file_first_second.write(str(",".join(features_first_second)) + "}\n\n@data\n")

    if args.token is not None:
        arff_file_first.write(str(",".join(features_first)) + "}\n\n@data\n")
        arff_file_second.write(str(",".join(features_second)) + "}\n\n@data\n")

    for i in range(n_docs):
        arff_file_first_second.write("{")

        if args.token is not None:
            arff_file_first.write("{")
            arff_file_second.write("{")

        for j in range(n_feats):
            if matrix[i, j] != 0:
                arff_file_first_second.write(str(j) + " " + str(matrix[i, j]) + ", ")

                if args.token is not None:
                    arff_file_first.write(str(j) + " " + str(matrix[i, j]) + ", ")
                    arff_file_second.write(str(j) + " " + str(matrix[i, j]) + ", ")

        arff_file_first_second.write(str(n_feats) + " " + str(features_first_second[class_atr_first_second_i[i]]) + "}\n")

        if args.token is not None:
            arff_file_first.write(str(n_feats) + " " + str(features_first[class_atr_first_i[i]]) + "}\n")
            arff_file_second.write(str(n_feats) + " " + str(features_second[class_atr_second_i[i]]) + "}\n")

    file_item.close()
    arff_file_first_second.close()

    if args.token is not None:
        arff_file_first.close()
        arff_file_second.close()

    filepath_i += 1
    end = time.time()
    eta = (total_num_examples-filepath_i)*(end-start)
    print_progress(filepath_i, total_num_examples, eta)

operation_end = time.time()
eta = operation_end-operation_start
print_progress(total_num_examples, total_num_examples, eta, final=True)
print("..................................................\n\n\n")

################################################################################


################################################################################

total_end = time.time()
print("> Log:")
print("..................................................")
print("- Time: " + str(format_time(total_end-total_start)))
print("- Output files: " + str(total_num_examples*3))
print("..................................................\n")