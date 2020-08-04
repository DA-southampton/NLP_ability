#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import six
import argparse
import torch
from onmt.utils.logging import init_logger, logger
from onmt.inputters.inputter import _old_style_vocab


def get_vocabs(dict_path):
    fields = torch.load(dict_path)

    vocs = []
    for side in ['src', 'tgt']:
        if _old_style_vocab(fields):
            vocab = next((v for n, v in fields if n == side), None)
        else:
            try:
                vocab = fields[side].base_field.vocab
            except AttributeError:
                vocab = fields[side].vocab
        vocs.append(vocab)
    enc_vocab, dec_vocab = vocs

    logger.info("From: %s" % dict_path)
    logger.info("\t* source vocab: %d words" % len(enc_vocab))
    logger.info("\t* target vocab: %d words" % len(dec_vocab))

    return enc_vocab, dec_vocab


def read_embeddings(file_enc, skip_lines=0, filter_set=None):
    embs = dict()
    total_vectors_in_file = 0
    with open(file_enc, 'rb') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            l_split = line.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            total_vectors_in_file += 1
            if filter_set is not None and l_split[0] not in filter_set:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
    return embs, total_vectors_in_file


def convert_to_torch_tensor(word_to_float_list_dict, vocab):
    dim = len(six.next(six.itervalues(word_to_float_list_dict)))
    tensor = torch.zeros((len(vocab), dim))
    for word, values in word_to_float_list_dict.items():
        tensor[vocab.stoi[word]] = torch.Tensor(values)
    return tensor


def calc_vocab_load_stats(vocab, loaded_embed_dict):
    matching_count = len(
        set(vocab.stoi.keys()) & set(loaded_embed_dict.keys()))
    missing_count = len(vocab) - matching_count
    percent_matching = matching_count / len(vocab) * 100
    return matching_count, missing_count, percent_matching


def main():
    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('-emb_file_both', required=False,
                        help="loads Embeddings for both source and target "
                             "from this file.")
    parser.add_argument('-emb_file_enc', required=False,
                        help="source Embeddings from this file")
    parser.add_argument('-emb_file_dec', required=False,
                        help="target Embeddings from this file")
    parser.add_argument('-output_file', required=True,
                        help="Output file for the prepared data")
    parser.add_argument('-dict_file', required=True,
                        help="Dictionary file")
    parser.add_argument('-verbose', action="store_true", default=False)
    parser.add_argument('-skip_lines', type=int, default=0,
                        help="Skip first lines of the embedding file")
    parser.add_argument('-type', choices=["GloVe", "word2vec"],
                        default="GloVe")
    opt = parser.parse_args()

    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)

    # Read in embeddings
    skip_lines = 1 if opt.type == "word2vec" else opt.skip_lines
    if opt.emb_file_both is not None:
        if opt.emb_file_enc is not None:
            raise ValueError("If --emb_file_both is passed in, you should not"
                             "set --emb_file_enc.")
        if opt.emb_file_dec is not None:
            raise ValueError("If --emb_file_both is passed in, you should not"
                             "set --emb_file_dec.")
        set_of_src_and_tgt_vocab = \
            set(enc_vocab.stoi.keys()) | set(dec_vocab.stoi.keys())
        logger.info("Reading encoder and decoder embeddings from {}".format(
            opt.emb_file_both))
        src_vectors, total_vec_count = \
            read_embeddings(opt.emb_file_both, skip_lines,
                            set_of_src_and_tgt_vocab)
        tgt_vectors = src_vectors
        logger.info("\tFound {} total vectors in file".format(total_vec_count))
    else:
        if opt.emb_file_enc is None:
            raise ValueError("If --emb_file_enc not provided. Please specify "
                             "the file with encoder embeddings, or pass in "
                             "--emb_file_both")
        if opt.emb_file_dec is None:
            raise ValueError("If --emb_file_dec not provided. Please specify "
                             "the file with encoder embeddings, or pass in "
                             "--emb_file_both")
        logger.info("Reading encoder embeddings from {}".format(
            opt.emb_file_enc))
        src_vectors, total_vec_count = read_embeddings(
            opt.emb_file_enc, skip_lines,
            filter_set=enc_vocab.stoi
        )
        logger.info("\tFound {} total vectors in file.".format(
            total_vec_count))
        logger.info("Reading decoder embeddings from {}".format(
            opt.emb_file_dec))
        tgt_vectors, total_vec_count = read_embeddings(
            opt.emb_file_dec, skip_lines,
            filter_set=dec_vocab.stoi
        )
        logger.info("\tFound {} total vectors in file".format(total_vec_count))
    logger.info("After filtering to vectors in vocab:")
    logger.info("\t* enc: %d match, %d missing, (%.2f%%)"
                % calc_vocab_load_stats(enc_vocab, src_vectors))
    logger.info("\t* dec: %d match, %d missing, (%.2f%%)"
                % calc_vocab_load_stats(dec_vocab, tgt_vectors))

    # Write to file
    enc_output_file = opt.output_file + ".enc.pt"
    dec_output_file = opt.output_file + ".dec.pt"
    logger.info("\nSaving embedding as:\n\t* enc: %s\n\t* dec: %s"
                % (enc_output_file, dec_output_file))
    torch.save(
        convert_to_torch_tensor(src_vectors, enc_vocab),
        enc_output_file
    )
    torch.save(
        convert_to_torch_tensor(tgt_vectors, dec_vocab),
        dec_output_file
    )
    logger.info("\nDone.")


if __name__ == "__main__":
    init_logger('embeddings_to_torch.log')
    main()
