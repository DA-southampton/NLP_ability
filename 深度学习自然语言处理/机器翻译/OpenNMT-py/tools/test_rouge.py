# -*- encoding: utf-8 -*-
import argparse
import os
import time
import pyrouge
import shutil
import sys
import codecs

from onmt.utils.logging import init_logger, logger


def test_rouge(cand, ref):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    try:
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
            os.mkdir(tmp_dir + "/candidate")
            os.mkdir(tmp_dir + "/reference")
        candidates = [line.strip() for line in cand]
        references = [line.strip() for line in ref]
        assert len(candidates) == len(references)
        cnt = len(candidates)
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        return results_dict
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def rouge_results_to_str(results_dict):
    return ">> ROUGE(1/2/3/L/SU4): {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_su*_f_score"] * 100)


if __name__ == "__main__":
    init_logger('test_rouge.log')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="candidate.txt",
                        help='candidate file')
    parser.add_argument('-r', type=str, default="reference.txt",
                        help='reference file')
    args = parser.parse_args()
    if args.c.upper() == "STDIN":
        candidates = sys.stdin
    else:
        candidates = codecs.open(args.c, encoding="utf-8")
    references = codecs.open(args.r, encoding="utf-8")

    results_dict = test_rouge(candidates, references)
    logger.info(rouge_results_to_str(results_dict))
