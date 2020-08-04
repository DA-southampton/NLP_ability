#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import sys
import os


def read_files_batch(file_list):
    """Reads the provided files in batches"""
    batch = []  # Keep batch for each file
    fd_list = []  # File descriptor list

    exit = False  # Flag used for quitting the program in case of error
    try:
        for filename in file_list:
            fd_list.append(open(filename))

        for lines in zip(*fd_list):
            for i, line in enumerate(lines):
                line = line.rstrip("\n").split(" ")
                batch.append(line)

            yield batch
            batch = []  # Reset batch

    except IOError:
        print("Error reading file " + filename + ".")
        exit = True  # Flag to exit the program

    finally:
        for fd in fd_list:
            fd.close()

        if exit:  # An error occurred, end execution
            sys.exit(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_type', default='text',
                        choices=['text', 'field'], required=True,
                        help="""Options for vocabulary creation.
                               The default is 'text' where the user passes
                               a corpus or a list of corpora files for which
                               they want to create a vocabulary from.
                               If choosing the option 'field', we assume
                               the file passed is a torch file created during
                               the preprocessing stage of an already
                               preprocessed corpus. The vocabulary file created
                               will just be the vocabulary inside the field
                               corresponding to the argument 'side'.""")
    parser.add_argument("-file", type=str, nargs="+", required=True)
    parser.add_argument("-out_file", type=str, required=True)
    parser.add_argument("-side", choices=['src', 'tgt'], help="""Specifies
                               'src' or 'tgt' side for 'field' file_type.""")

    opt = parser.parse_args()

    vocabulary = {}
    if opt.file_type == 'text':
        print("Reading input file...")
        for batch in read_files_batch(opt.file):
            for sentence in batch:
                for w in sentence:
                    if w in vocabulary:
                        vocabulary[w] += 1
                    else:
                        vocabulary[w] = 1

        print("Writing vocabulary file...")
        with open(opt.out_file, "w") as f:
            for w, count in sorted(vocabulary.items(), key=lambda x: x[1],
                                   reverse=True):
                f.write("{0}\n".format(w))
    else:
        if opt.side not in ['src', 'tgt']:
            raise ValueError("If using -file_type='field', specifies "
                             "'src' or 'tgt' argument for -side.")
        import torch
        try:
            from onmt.inputters.inputter import _old_style_vocab
        except ImportError:
            sys.path.insert(1, os.path.join(sys.path[0], '..'))
            from onmt.inputters.inputter import _old_style_vocab

        print("Reading input file...")
        if not len(opt.file) == 1:
            raise ValueError("If using -file_type='field', only pass one "
                             "argument for -file.")
        vocabs = torch.load(opt.file[0])
        voc = dict(vocabs)[opt.side]
        if _old_style_vocab(voc):
            word_list = voc.itos
        else:
            try:
                word_list = voc[0][1].base_field.vocab.itos
            except AttributeError:
                word_list = voc[0][1].vocab.itos

        print("Writing vocabulary file...")
        with open(opt.out_file, "wb") as f:
            for w in word_list:
                f.write(u"{0}\n".format(w).encode("utf-8"))


if __name__ == "__main__":
    main()
