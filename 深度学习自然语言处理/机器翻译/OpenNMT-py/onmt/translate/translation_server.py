#!/usr/bin/env python
"""REST Translation server."""
from __future__ import print_function
import codecs
import sys
import os
import time
import json
import threading
import re
import traceback
import importlib
import torch
import onmt.opts

from onmt.utils.logging import init_logger
from onmt.utils.misc import set_random_seed
from onmt.utils.misc import check_model_config
from onmt.utils.alignment import to_word_align
from onmt.utils.parse import ArgumentParser
from onmt.translate.translator import build_translator


def critical(func):
    """Decorator for critical section (mutually exclusive code)"""
    def wrapper(server_model, *args, **kwargs):
        if sys.version_info[0] == 3:
            if not server_model.running_lock.acquire(True, 120):
                raise ServerModelError("Model %d running lock timeout"
                                       % server_model.model_id)
        else:
            # semaphore doesn't have a timeout arg in Python 2.7
            server_model.running_lock.acquire(True)
        try:
            o = func(server_model, *args, **kwargs)
        except (Exception, RuntimeError):
            server_model.running_lock.release()
            raise
        server_model.running_lock.release()
        return o
    return wrapper


class Timer:
    def __init__(self, start=False):
        self.stime = -1
        self.prev = -1
        self.times = {}
        if start:
            self.start()

    def start(self):
        self.stime = time.time()
        self.prev = self.stime
        self.times = {}

    def tick(self, name=None, tot=False):
        t = time.time()
        if not tot:
            elapsed = t - self.prev
        else:
            elapsed = t - self.stime
        self.prev = t

        if name is not None:
            self.times[name] = elapsed
        return elapsed


class ServerModelError(Exception):
    pass


class TranslationServer(object):
    def __init__(self):
        self.models = {}
        self.next_id = 0

    def start(self, config_file):
        """Read the config file and pre-/load the models."""
        self.config_file = config_file
        with open(self.config_file) as f:
            self.confs = json.load(f)

        self.models_root = self.confs.get('models_root', './available_models')
        for i, conf in enumerate(self.confs["models"]):
            if "models" not in conf:
                if "model" in conf:
                    # backwards compatibility for confs
                    conf["models"] = [conf["model"]]
                else:
                    raise ValueError("""Incorrect config file: missing 'models'
                                        parameter for model #%d""" % i)
            check_model_config(conf, self.models_root)
            kwargs = {'timeout': conf.get('timeout', None),
                      'load': conf.get('load', None),
                      'preprocess_opt': conf.get('preprocess', None),
                      'tokenizer_opt': conf.get('tokenizer', None),
                      'postprocess_opt': conf.get('postprocess', None),
                      'on_timeout': conf.get('on_timeout', None),
                      'model_root': conf.get('model_root', self.models_root)
                      }
            kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
            model_id = conf.get("id", None)
            opt = conf["opt"]
            opt["models"] = conf["models"]
            self.preload_model(opt, model_id=model_id, **kwargs)

    def clone_model(self, model_id, opt, timeout=-1):
        """Clone a model `model_id`.

        Different options may be passed. If `opt` is None, it will use the
        same set of options
        """
        if model_id in self.models:
            if opt is None:
                opt = self.models[model_id].user_opt
            opt["models"] = self.models[model_id].opt.models
            return self.load_model(opt, timeout)
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def load_model(self, opt, model_id=None, **model_kwargs):
        """Load a model given a set of options
        """
        model_id = self.preload_model(opt, model_id=model_id, **model_kwargs)
        load_time = self.models[model_id].load_time

        return model_id, load_time

    def preload_model(self, opt, model_id=None, **model_kwargs):
        """Preloading the model: updating internal datastructure

        It will effectively load the model if `load` is set
        """
        if model_id is not None:
            if model_id in self.models.keys():
                raise ValueError("Model ID %d already exists" % model_id)
        else:
            model_id = self.next_id
            while model_id in self.models.keys():
                model_id += 1
            self.next_id = model_id + 1
        print("Pre-loading model %d" % model_id)
        model = ServerModel(opt, model_id, **model_kwargs)
        self.models[model_id] = model

        return model_id

    def run(self, inputs):
        """Translate `inputs`

        We keep the same format as the Lua version i.e.
        ``[{"id": model_id, "src": "sequence to translate"},{ ...}]``

        We use inputs[0]["id"] as the model id
        """

        model_id = inputs[0].get("id", 0)
        if model_id in self.models and self.models[model_id] is not None:
            return self.models[model_id].run(inputs)
        else:
            print("Error No such model '%s'" % str(model_id))
            raise ServerModelError("No such model '%s'" % str(model_id))

    def unload_model(self, model_id):
        """Manually unload a model.

        It will free the memory and cancel the timer
        """

        if model_id in self.models and self.models[model_id] is not None:
            self.models[model_id].unload()
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def list_models(self):
        """Return the list of available models
        """
        models = []
        for _, model in self.models.items():
            models += [model.to_dict()]
        return models


class ServerModel(object):
    """Wrap a model with server functionality.

    Args:
        opt (dict): Options for the Translator
        model_id (int): Model ID
        preprocess_opt (list): Options for preprocess processus or None
                               (extend for CJK)
        tokenizer_opt (dict): Options for the tokenizer or None
        postprocess_opt (list): Options for postprocess processus or None
                                (extend for CJK)
        load (bool): whether to load the model during :func:`__init__()`
        timeout (int): Seconds before running :func:`do_timeout()`
            Negative values means no timeout
        on_timeout (str): Options are ["to_cpu", "unload"]. Set what to do on
            timeout (see :func:`do_timeout()`.)
        model_root (str): Path to the model directory
            it must contain the model and tokenizer file
    """

    def __init__(self, opt, model_id, preprocess_opt=None, tokenizer_opt=None,
                 postprocess_opt=None, load=False, timeout=-1,
                 on_timeout="to_cpu", model_root="./"):
        self.model_root = model_root
        self.opt = self.parse_opt(opt)

        self.model_id = model_id
        self.preprocess_opt = preprocess_opt
        self.tokenizer_opt = tokenizer_opt
        self.postprocess_opt = postprocess_opt
        self.timeout = timeout
        self.on_timeout = on_timeout

        self.unload_timer = None
        self.user_opt = opt
        self.tokenizer = None

        if len(self.opt.log_file) > 0:
            log_file = os.path.join(model_root, self.opt.log_file)
        else:
            log_file = None
        self.logger = init_logger(log_file=log_file,
                                  log_file_level=self.opt.log_file_level)

        self.loading_lock = threading.Event()
        self.loading_lock.set()
        self.running_lock = threading.Semaphore(value=1)

        set_random_seed(self.opt.seed, self.opt.cuda)

        if load:
            self.load()

    def parse_opt(self, opt):
        """Parse the option set passed by the user using `onmt.opts`

       Args:
           opt (dict): Options passed by the user

       Returns:
           opt (argparse.Namespace): full set of options for the Translator
        """

        prec_argv = sys.argv
        sys.argv = sys.argv[:1]
        parser = ArgumentParser()
        onmt.opts.translate_opts(parser)

        models = opt['models']
        if not isinstance(models, (list, tuple)):
            models = [models]
        opt['models'] = [os.path.join(self.model_root, model)
                         for model in models]
        opt['src'] = "dummy_src"

        for (k, v) in opt.items():
            if k == 'models':
                sys.argv += ['-model']
                sys.argv += [str(model) for model in v]
            elif type(v) == bool:
                sys.argv += ['-%s' % k]
            else:
                sys.argv += ['-%s' % k, str(v)]

        opt = parser.parse_args()
        ArgumentParser.validate_translate_opts(opt)
        opt.cuda = opt.gpu > -1

        sys.argv = prec_argv
        return opt

    @property
    def loaded(self):
        return hasattr(self, 'translator')

    def load(self):
        self.loading_lock.clear()

        timer = Timer()
        self.logger.info("Loading model %d" % self.model_id)
        timer.start()

        try:
            self.translator = build_translator(self.opt,
                                               report_score=False,
                                               out_file=codecs.open(
                                                   os.devnull, "w", "utf-8"))
        except RuntimeError as e:
            raise ServerModelError("Runtime Error: %s" % str(e))

        timer.tick("model_loading")
        if self.preprocess_opt is not None:
            self.logger.info("Loading preprocessor")
            self.preprocessor = []

            for function_path in self.preprocess_opt:
                function = get_function_by_path(function_path)
                self.preprocessor.append(function)

        if self.tokenizer_opt is not None:
            self.logger.info("Loading tokenizer")

            if "type" not in self.tokenizer_opt:
                raise ValueError(
                    "Missing mandatory tokenizer option 'type'")

            if self.tokenizer_opt['type'] == 'sentencepiece':
                if "model" not in self.tokenizer_opt:
                    raise ValueError(
                        "Missing mandatory tokenizer option 'model'")
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor()
                model_path = os.path.join(self.model_root,
                                          self.tokenizer_opt['model'])
                sp.Load(model_path)
                self.tokenizer = sp
            elif self.tokenizer_opt['type'] == 'pyonmttok':
                if "params" not in self.tokenizer_opt:
                    raise ValueError(
                        "Missing mandatory tokenizer option 'params'")
                import pyonmttok
                if self.tokenizer_opt["mode"] is not None:
                    mode = self.tokenizer_opt["mode"]
                else:
                    mode = None
                # load can be called multiple times: modify copy
                tokenizer_params = dict(self.tokenizer_opt["params"])
                for key, value in self.tokenizer_opt["params"].items():
                    if key.endswith("path"):
                        tokenizer_params[key] = os.path.join(
                            self.model_root, value)
                tokenizer = pyonmttok.Tokenizer(mode,
                                                **tokenizer_params)
                self.tokenizer = tokenizer
            else:
                raise ValueError("Invalid value for tokenizer type")

        if self.postprocess_opt is not None:
            self.logger.info("Loading postprocessor")
            self.postprocessor = []

            for function_path in self.postprocess_opt:
                function = get_function_by_path(function_path)
                self.postprocessor.append(function)

        self.load_time = timer.tick()
        self.reset_unload_timer()
        self.loading_lock.set()

    @critical
    def run(self, inputs):
        """Translate `inputs` using this model

        Args:
            inputs (List[dict[str, str]]): [{"src": "..."},{"src": ...}]

        Returns:
            result (list): translations
            times (dict): containing times
        """

        self.stop_unload_timer()

        timer = Timer()
        timer.start()

        self.logger.info("Running translation using %d" % self.model_id)

        if not self.loading_lock.is_set():
            self.logger.info(
                "Model #%d is being loaded by another thread, waiting"
                % self.model_id)
            if not self.loading_lock.wait(timeout=30):
                raise ServerModelError("Model %d loading timeout"
                                       % self.model_id)

        else:
            if not self.loaded:
                self.load()
                timer.tick(name="load")
            elif self.opt.cuda:
                self.to_gpu()
                timer.tick(name="to_gpu")

        texts = []
        head_spaces = []
        tail_spaces = []
        sslength = []
        for i, inp in enumerate(inputs):
            src = inp['src']
            if src.strip() == "":
                head_spaces.append(src)
                texts.append("")
                tail_spaces.append("")
            else:
                whitespaces_before, whitespaces_after = "", ""
                match_before = re.search(r'^\s+', src)
                match_after = re.search(r'\s+$', src)
                if match_before is not None:
                    whitespaces_before = match_before.group(0)
                if match_after is not None:
                    whitespaces_after = match_after.group(0)
                head_spaces.append(whitespaces_before)
                preprocessed_src = self.maybe_preprocess(src.strip())
                tok = self.maybe_tokenize(preprocessed_src)
                texts.append(tok)
                sslength.append(len(tok.split()))
                tail_spaces.append(whitespaces_after)

        empty_indices = [i for i, x in enumerate(texts) if x == ""]
        texts_to_translate = [x for x in texts if x != ""]

        scores = []
        predictions = []
        if len(texts_to_translate) > 0:
            try:
                scores, predictions = self.translator.translate(
                    texts_to_translate,
                    batch_size=len(texts_to_translate)
                    if self.opt.batch_size == 0
                    else self.opt.batch_size)
            except (RuntimeError, Exception) as e:
                err = "Error: %s" % str(e)
                self.logger.error(err)
                self.logger.error("repr(text_to_translate): "
                                  + repr(texts_to_translate))
                self.logger.error("model: #%s" % self.model_id)
                self.logger.error("model opt: " + str(self.opt.__dict__))
                self.logger.error(traceback.format_exc())

                raise ServerModelError(err)

        timer.tick(name="translation")
        self.logger.info("""Using model #%d\t%d inputs
               \ttranslation time: %f""" % (self.model_id, len(texts),
                                            timer.times['translation']))
        self.reset_unload_timer()

        # NOTE: translator returns lists of `n_best` list
        def flatten_list(_list): return sum(_list, [])
        tiled_texts = [t for t in texts_to_translate
                       for _ in range(self.opt.n_best)]
        results = flatten_list(predictions)
        scores = [score_tensor.item()
                  for score_tensor in flatten_list(scores)]

        results = [self.maybe_detokenize_with_align(result, src)
                   for result, src in zip(results, tiled_texts)]

        aligns = [align for _, align in results]
        results = [self.maybe_postprocess(seq) for seq, _ in results]

        # build back results with empty texts
        for i in empty_indices:
            j = i * self.opt.n_best
            results = results[:j] + [""] * self.opt.n_best + results[j:]
            aligns = aligns[:j] + [None] * self.opt.n_best + aligns[j:]
            scores = scores[:j] + [0] * self.opt.n_best + scores[j:]

        head_spaces = [h for h in head_spaces for i in range(self.opt.n_best)]
        tail_spaces = [h for h in tail_spaces for i in range(self.opt.n_best)]
        results = ["".join(items)
                   for items in zip(head_spaces, results, tail_spaces)]

        self.logger.info("Translation Results: %d", len(results))
        return results, scores, self.opt.n_best, timer.times, aligns

    def do_timeout(self):
        """Timeout function that frees GPU memory.

        Moves the model to CPU or unloads it; depending on
        attr`self.on_timemout` value
        """

        if self.on_timeout == "unload":
            self.logger.info("Timeout: unloading model %d" % self.model_id)
            self.unload()
        if self.on_timeout == "to_cpu":
            self.logger.info("Timeout: sending model %d to CPU"
                             % self.model_id)
            self.to_cpu()

    @critical
    def unload(self):
        self.logger.info("Unloading model %d" % self.model_id)
        del self.translator
        if self.opt.cuda:
            torch.cuda.empty_cache()
        self.unload_timer = None

    def stop_unload_timer(self):
        if self.unload_timer is not None:
            self.unload_timer.cancel()

    def reset_unload_timer(self):
        if self.timeout < 0:
            return

        self.stop_unload_timer()
        self.unload_timer = threading.Timer(self.timeout, self.do_timeout)
        self.unload_timer.start()

    def to_dict(self):
        hide_opt = ["models", "src"]
        d = {"model_id": self.model_id,
             "opt": {k: self.user_opt[k] for k in self.user_opt.keys()
                     if k not in hide_opt},
             "models": self.user_opt["models"],
             "loaded": self.loaded,
             "timeout": self.timeout,
             }
        if self.tokenizer_opt is not None:
            d["tokenizer"] = self.tokenizer_opt
        return d

    @critical
    def to_cpu(self):
        """Move the model to CPU and clear CUDA cache."""
        self.translator.model.cpu()
        if self.opt.cuda:
            torch.cuda.empty_cache()

    def to_gpu(self):
        """Move the model to GPU."""
        torch.cuda.set_device(self.opt.gpu)
        self.translator.model.cuda()

    def maybe_preprocess(self, sequence):
        """Preprocess the sequence (or not)

        """

        if self.preprocess_opt is not None:
            return self.preprocess(sequence)
        return sequence

    def preprocess(self, sequence):
        """Preprocess a single sequence.

        Args:
            sequence (str): The sequence to preprocess.

        Returns:
            sequence (str): The preprocessed sequence.
        """
        if self.preprocessor is None:
            raise ValueError("No preprocessor loaded")
        for function in self.preprocessor:
            sequence = function(sequence)
        return sequence

    def maybe_tokenize(self, sequence):
        """Tokenize the sequence (or not).

        Same args/returns as `tokenize`
        """

        if self.tokenizer_opt is not None:
            return self.tokenize(sequence)
        return sequence

    def tokenize(self, sequence):
        """Tokenize a single sequence.

        Args:
            sequence (str): The sequence to tokenize.

        Returns:
            tok (str): The tokenized sequence.
        """

        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded")

        if self.tokenizer_opt["type"] == "sentencepiece":
            tok = self.tokenizer.EncodeAsPieces(sequence)
            tok = " ".join(tok)
        elif self.tokenizer_opt["type"] == "pyonmttok":
            tok, _ = self.tokenizer.tokenize(sequence)
            tok = " ".join(tok)
        return tok

    @property
    def tokenizer_marker(self):
        marker = None
        tokenizer_type = self.tokenizer_opt.get('type', None)
        if tokenizer_type == "pyonmttok":
            params = self.tokenizer_opt.get('params', None)
            if params is not None:
                if params.get("joiner_annotate", None) is not None:
                    marker = 'joiner'
                elif params.get("spacer_annotate", None) is not None:
                    marker = 'spacer'
        elif tokenizer_type == "sentencepiece":
            marker = 'spacer'
        return marker

    def maybe_detokenize_with_align(self, sequence, src):
        """De-tokenize (or not) the sequence (with alignment).

        Args:
            sequence (str): The sequence to detokenize, possible with
                alignment seperate by ` ||| `.

        Returns:
            sequence (str): The detokenized sequence.
            align (str): The alignment correspand to detokenized src/tgt
                sorted or None if no alignment in output.
        """
        align = None
        if self.opt.report_align:
            # output contain alignment
            sequence, align = sequence.split(' ||| ')
            align = self.maybe_convert_align(src, sequence, align)
        sequence = self.maybe_detokenize(sequence)
        return (sequence, align)

    def maybe_detokenize(self, sequence):
        """De-tokenize the sequence (or not)

        Same args/returns as :func:`tokenize()`
        """

        if self.tokenizer_opt is not None and ''.join(sequence.split()) != '':
            return self.detokenize(sequence)
        return sequence

    def detokenize(self, sequence):
        """Detokenize a single sequence

        Same args/returns as :func:`tokenize()`
        """

        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded")

        if self.tokenizer_opt["type"] == "sentencepiece":
            detok = self.tokenizer.DecodePieces(sequence.split())
        elif self.tokenizer_opt["type"] == "pyonmttok":
            detok = self.tokenizer.detokenize(sequence.split())

        return detok

    def maybe_convert_align(self, src, tgt, align):
        """Convert alignment to match detokenized src/tgt (or not).

        Args:
            src (str): The tokenized source sequence.
            tgt (str): The tokenized target sequence.
            align (str): The alignment correspand to src/tgt pair.

        Returns:
            align (str): The alignment correspand to detokenized src/tgt.
        """
        if self.tokenizer_marker is not None and ''.join(tgt.split()) != '':
            return to_word_align(src, tgt, align, mode=self.tokenizer_marker)
        return align

    def maybe_postprocess(self, sequence):
        """Postprocess the sequence (or not)

        """

        if self.postprocess_opt is not None:
            return self.postprocess(sequence)
        return sequence

    def postprocess(self, sequence):
        """Preprocess a single sequence.

        Args:
            sequence (str): The sequence to process.

        Returns:
            sequence (str): The postprocessed sequence.
        """
        if self.postprocessor is None:
            raise ValueError("No postprocessor loaded")
        for function in self.postprocessor:
            sequence = function(sequence)
        return sequence


def get_function_by_path(path, args=[], kwargs={}):
    module_name = ".".join(path.split(".")[:-1])
    function_name = path.split(".")[-1]
    try:
        module = importlib.import_module(module_name)
    except ValueError as e:
        print("Cannot import module '%s'" % module_name)
        raise e
    function = getattr(module, function_name)
    return function
