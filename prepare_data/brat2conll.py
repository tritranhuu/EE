from __future__ import print_function

import os
import re
import sys
from collections import namedtuple
from io import StringIO
from os import path

path_data = 'data/test'
path_save = 'data/full'

def parse_textbounds(f, annfn):
    """Parse textbound annotations in input, returning a list of Textbound."""

    textbounds = dict()
    events = []
    for l in f:
        if l =='':
            continue
        l_split = l.split('\t')
        try:
            if l_split[0][0] =='T':
                id_, type_offsets, text = l_split
                pos = type_offsets.split()
                if len(pos)>3:
                    type_, start, _, end = pos
                else:
                    type_, start, end = pos

                start, end = int(start), int(end)
                if len(text) > end - start-5:
                    textbounds.update({id_: [start, end, type_, text]})
            else:
                events.append(l_split[1])
        except:
            pass
    for event in events:
        event_ = event.split()
        args = []
        for arg in event_[1:]:
            args.append(arg)

        textbounds[event_[0].split(':')[1]].append(args)
        textbounds[event_[0].split(':')[1]].append('E')

    return textbounds

def eliminate_overlaps(textbounds):
    eliminate = {}

    # TODO: avoid O(n^2) overlap check
    for k1 in textbounds:
        for k2 in textbounds:
            t1 = textbounds[k1]
            t2 = textbounds[k2]
            if t1 is t2 or len(t1)!= len(t2):
                continue
            if t2[0] >= t1[1] or t2[1] <= t1[0] :
                continue
            # eliminate shorter
            if t1[1]- t1[0] > t2[0] - t2[1]:
                print("Eliminate %s due to overlap with %s" % (
                    t2, t1), file=sys.stderr)
                eliminate[k2] = True
            else:
                print("Eliminate %s due to overlap with %s" % (
                    t1, t2), file=sys.stderr)
                eliminate[k1] = True

    return dict({k:textbounds[k] for k in textbounds if k not in eliminate})

def get_annotations(annfn):
    with open(annfn, 'rU', encoding='utf8') as f:
        f = f.read().split('\n')
        textbounds = parse_textbounds(f, annfn)

    # textbounds = eliminate_overlaps(textbounds)
    # print(textbounds)
    return textbounds

def text_to_conll(ftext, path_s):
    """Convert plain text into CoNLL format."""
    fspl = ftext.split('/')
    fann = ftext[: -len(fspl[-1])] + fspl[-1].split('.')[0] + '.ann'
    with open(ftext, encoding='utf8') as f:
        sentences = f.read().split('\n')
    lines = []

    offset = 0
    for s in sentences:
        if s =='':
            offset +=1
            continue
        tokens = s.split()
        line = []
        for t in tokens:
            if not t.isspace():
                line.append([t, offset, offset + len(t)])

            offset += len(t) +1
        lines.append(line)
    # print(lines)
    # print('----------------------')
    # add labels (other than 'O') from standoff annotation if specified
    lines = relabel(lines, get_annotations(fann))
    # print('----------------------')
    # print(lines)
    save_conll(lines, path_s)
    return lines

def relabel(lines, annotations):
    labedlines = []
    for l in lines:
        pre_ner, pre_event = None, None
        temp_line = []
        for token in l:
            text, start, end = token
            ntoken = [text, 'O', 'O', 'O']
            for T in annotations:
                infor = annotations[T]
                if text in infor[3] and start>= infor[0] and end <= infor[1]:
                    if len(infor) >4:
                        if pre_event is None or T != pre_event:
                            ntoken[1] = infor[2]
                            pre_event = T
                        else:
                            ntoken[1] = infor[2]

                        ntoken[3] = T
                        ntoken.append(infor[4])

                    else:
                        if pre_ner is None or T != pre_ner:
                            ntoken[2] =  infor[2]
                            ntoken[3] = T
                            pre_ner = T
                        else:
                            ntoken[2] =  infor[2]
                            ntoken[3] = T

            temp_line.append(ntoken)
        labedlines.append(temp_line)

    return labedlines

def save_conll(data, path, opt='w'):
    with open(path, opt, encoding='utf-8') as f:
        for sent in data:
            for word in sent:
                if len(word) ==4:
                    f.write(word[0] + '\t' + word[1] + '\t' + word[2] + '\t'+ word[3] + '\t' + 'N' + '\n')
                else:
                    f.write(word[0] + '\t' + word[1] + '\t' + word[2] + '\t'+ word[3] + '\t')
                    for arg in word[4]:
                        f.write(arg + ' ')
                    f.write('\n')
            f.write('\n')


# text_to_conll('C:/Users/dell/Downloads/Compressed/brat-v1.3_Crunchy_Frog/data/event/Phase_2/Bankruptcy/Bankruptcy_1/Bankruptcy_009.txt', './')
def createFolder(path, pathsave):
    folders = os.listdir(path)
    path_file = []
    paths_file = []
    for fo in folders:
        if fo in ['.stats_cache','annotation.conf','tools.conf','conllallinone.txt','kb_shortcuts.conf', 'output.ann', 'output.txt','visual.conf']:
            continue
        if os.path.isdir(path +'/'+ fo):
            path_fo = path +'/' + fo
            paths_fo = pathsave +'/' + fo
            try:
                os.makedirs(paths_fo)
            except:
                pass
            returnpaths = createFolder(path_fo, paths_fo)

            if len(returnpaths[0]):
                path_file.extend(returnpaths[0])
                paths_file.extend(returnpaths[1])
        else:
            if fo[-3:] =='ann':
                continue
            else:
                path_file.append(path + '/' + fo)
                paths_file.append(pathsave + '/' + fo)

    return  path_file, paths_file


def brat2conll():
    npath_file, npath_save = createFolder(path_data, path_save)
    total = []
    count = 0
    for file, files in zip(npath_file, npath_save):
        # print(file)
        save_conll(text_to_conll(file, files), path_save + '/test.txt', opt='a')
        count +=1
        if count %100 ==0:
            print('check {} files'.format(count))

brat2conll()