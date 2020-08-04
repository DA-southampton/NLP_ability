import argparse

import torch

import onmt
import onmt.model_builder
import onmt.inputters as inputters
import onmt.opts

from onmt.utils.misc import use_gpu
from onmt.utils.logging import init_logger, logger

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-output_dir', default='.',
                    help="""Path to output the embeddings""")
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def write_embeddings(filename, dict, embeddings):
    with open(filename, 'wb') as file:
        for i in range(min(len(embeddings), len(dict.itos))):
            str = dict.itos[i].encode("utf-8")
            for j in range(len(embeddings[0])):
                str = str + (" %5f" % (embeddings[i][j])).encode("utf-8")
            file.write(str + b"\n")


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    onmt.opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Add in default model arguments, possibly added since training.
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    model_opt = checkpoint['opt']

    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = onmt.inputters.load_old_vocab(vocab)
    else:
        fields = vocab
    src_dict = fields['src'].base_field.vocab  # assumes src is text
    tgt_dict = fields['tgt'].base_field.vocab

    model_opt = checkpoint['opt']
    for arg in dummy_opt.__dict__:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt.__dict__[arg]

    model = onmt.model_builder.build_base_model(
        model_opt, fields, use_gpu(opt), checkpoint)
    encoder = model.encoder
    decoder = model.decoder

    encoder_embeddings = encoder.embeddings.word_lut.weight.data.tolist()
    decoder_embeddings = decoder.embeddings.word_lut.weight.data.tolist()

    logger.info("Writing source embeddings")
    write_embeddings(opt.output_dir + "/src_embeddings.txt", src_dict,
                     encoder_embeddings)

    logger.info("Writing target embeddings")
    write_embeddings(opt.output_dir + "/tgt_embeddings.txt", tgt_dict,
                     decoder_embeddings)

    logger.info('... done.')
    logger.info('Converting model...')


if __name__ == "__main__":
    init_logger('extract_embeddings.log')
    main()
