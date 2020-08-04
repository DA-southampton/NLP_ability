""" Implementation of all available options """
from __future__ import print_function

import configargparse
from onmt.models.sru import CheckSRU


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add('--src_word_vec_size', '-src_word_vec_size',
              type=int, default=500,
              help='Word embedding size for src.')
    group.add('--tgt_word_vec_size', '-tgt_word_vec_size',
              type=int, default=500,
              help='Word embedding size for tgt.')
    group.add('--word_vec_size', '-word_vec_size', type=int, default=-1,
              help='Word embedding size for src and tgt.')

    group.add('--share_decoder_embeddings', '-share_decoder_embeddings',
              action='store_true',
              help="Use a shared weight matrix for the input and "
                   "output word  embeddings in the decoder.")
    group.add('--share_embeddings', '-share_embeddings', action='store_true',
              help="Share the word embeddings between encoder "
                   "and decoder. Need to use shared dictionary for this "
                   "option.")
    group.add('--position_encoding', '-position_encoding', action='store_true',
              help="Use a sin to mark relative words positions. "
                   "Necessary for non-RNN style models.")

    group = parser.add_argument_group('Model-Embedding Features')
    group.add('--feat_merge', '-feat_merge', type=str, default='concat',
              choices=['concat', 'sum', 'mlp'],
              help="Merge action for incorporating features embeddings. "
                   "Options [concat|sum|mlp].")
    group.add('--feat_vec_size', '-feat_vec_size', type=int, default=-1,
              help="If specified, feature embedding sizes "
                   "will be set to this. Otherwise, feat_vec_exponent "
                   "will be used.")
    group.add('--feat_vec_exponent', '-feat_vec_exponent',
              type=float, default=0.7,
              help="If -feat_merge_size is not set, feature "
                   "embedding sizes will be set to N^feat_vec_exponent "
                   "where N is the number of values the feature takes.")

    # Encoder-Decoder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add('--model_type', '-model_type', default='text',
              choices=['text', 'img', 'audio', 'vec'],
              help="Type of source model to use. Allows "
                   "the system to incorporate non-text inputs. "
                   "Options are [text|img|audio|vec].")
    group.add('--model_dtype', '-model_dtype', default='fp32',
              choices=['fp32', 'fp16'],
              help='Data type of the model.')

    group.add('--encoder_type', '-encoder_type', type=str, default='rnn',
              choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
              help="Type of encoder layer to use. Non-RNN layers "
                   "are experimental. Options are "
                   "[rnn|brnn|mean|transformer|cnn].")
    group.add('--decoder_type', '-decoder_type', type=str, default='rnn',
              choices=['rnn', 'transformer', 'cnn'],
              help="Type of decoder layer to use. Non-RNN layers "
                   "are experimental. Options are "
                   "[rnn|transformer|cnn].")

    group.add('--layers', '-layers', type=int, default=-1,
              help='Number of layers in enc/dec.')
    group.add('--enc_layers', '-enc_layers', type=int, default=2,
              help='Number of layers in the encoder')
    group.add('--dec_layers', '-dec_layers', type=int, default=2,
              help='Number of layers in the decoder')
    group.add('--rnn_size', '-rnn_size', type=int, default=-1,
              help="Size of rnn hidden states. Overwrites "
                   "enc_rnn_size and dec_rnn_size")
    group.add('--enc_rnn_size', '-enc_rnn_size', type=int, default=500,
              help="Size of encoder rnn hidden states. "
                   "Must be equal to dec_rnn_size except for "
                   "speech-to-text.")
    group.add('--dec_rnn_size', '-dec_rnn_size', type=int, default=500,
              help="Size of decoder rnn hidden states. "
                   "Must be equal to enc_rnn_size except for "
                   "speech-to-text.")
    group.add('--audio_enc_pooling', '-audio_enc_pooling',
              type=str, default='1',
              help="The amount of pooling of audio encoder, "
                   "either the same amount of pooling across all layers "
                   "indicated by a single number, or different amounts of "
                   "pooling per layer separated by comma.")
    group.add('--cnn_kernel_width', '-cnn_kernel_width', type=int, default=3,
              help="Size of windows in the cnn, the kernel_size is "
                   "(cnn_kernel_width, 1) in conv layer")

    group.add('--input_feed', '-input_feed', type=int, default=1,
              help="Feed the context vector at each time step as "
                   "additional input (via concatenation with the word "
                   "embeddings) to the decoder.")
    group.add('--bridge', '-bridge', action="store_true",
              help="Have an additional layer between the last encoder "
                   "state and the first decoder state")
    group.add('--rnn_type', '-rnn_type', type=str, default='LSTM',
              choices=['LSTM', 'GRU', 'SRU'],
              action=CheckSRU,
              help="The gate type to use in the RNNs")
    # group.add('--residual', '-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    group.add('--brnn', '-brnn', action=DeprecateAction,
              help="Deprecated, use `encoder_type`.")

    group.add('--context_gate', '-context_gate', type=str, default=None,
              choices=['source', 'target', 'both'],
              help="Type of context gate to use. "
                   "Do not select for no context gate.")

    # Attention options
    group = parser.add_argument_group('Model- Attention')
    group.add('--global_attention', '-global_attention',
              type=str, default='general',
              choices=['dot', 'general', 'mlp', 'none'],
              help="The attention type to use: "
                   "dotprod or general (Luong) or MLP (Bahdanau)")
    group.add('--global_attention_function', '-global_attention_function',
              type=str, default="softmax", choices=["softmax", "sparsemax"])
    group.add('--self_attn_type', '-self_attn_type',
              type=str, default="scaled-dot",
              help='Self attention type in Transformer decoder '
                   'layer -- currently "scaled-dot" or "average" ')
    group.add('--max_relative_positions', '-max_relative_positions',
              type=int, default=0,
              help="Maximum distance between inputs in relative "
                   "positions representations. "
                   "For more detailed information, see: "
                   "https://arxiv.org/pdf/1803.02155.pdf")
    group.add('--heads', '-heads', type=int, default=8,
              help='Number of heads for transformer self-attention')
    group.add('--transformer_ff', '-transformer_ff', type=int, default=2048,
              help='Size of hidden transformer feed-forward')
    group.add('--aan_useffn', '-aan_useffn', action="store_true",
              help='Turn on the FFN layer in the AAN decoder')

    # Alignement options
    group = parser.add_argument_group('Model - Alignement')
    group.add('--lambda_align', '-lambda_align', type=float, default=0.0,
              help="Lambda value for alignement loss of Garg et al (2019)"
                   "For more detailed information, see: "
                   "https://arxiv.org/abs/1909.02074")
    group.add('--alignment_layer', '-alignment_layer', type=int, default=-3,
              help='Layer number which has to be supervised.')
    group.add('--alignment_heads', '-alignment_heads', type=int, default=None,
              help='N. of cross attention heads per layer to supervised with')
    group.add('--full_context_alignment', '-full_context_alignment',
              action="store_true",
              help='Whether alignment is conditioned on full target context.')

    # Generator and loss options.
    group = parser.add_argument_group('Generator')
    group.add('--copy_attn', '-copy_attn', action="store_true",
              help='Train copy attention layer.')
    group.add('--copy_attn_type', '-copy_attn_type',
              type=str, default=None,
              choices=['dot', 'general', 'mlp', 'none'],
              help="The copy attention type to use. Leave as None to use "
                   "the same as -global_attention.")
    group.add('--generator_function', '-generator_function', default="softmax",
              choices=["softmax", "sparsemax"],
              help="Which function to use for generating "
                   "probabilities over the target vocabulary (choices: "
                   "softmax, sparsemax)")
    group.add('--copy_attn_force', '-copy_attn_force', action="store_true",
              help='When available, train to copy.')
    group.add('--reuse_copy_attn', '-reuse_copy_attn', action="store_true",
              help="Reuse standard attention for copy")
    group.add('--copy_loss_by_seqlength', '-copy_loss_by_seqlength',
              action="store_true",
              help="Divide copy loss by length of sequence")
    group.add('--coverage_attn', '-coverage_attn', action="store_true",
              help='Train a coverage attention layer.')
    group.add('--lambda_coverage', '-lambda_coverage', type=float, default=0.0,
              help='Lambda value for coverage loss of See et al (2017)')
    group.add('--loss_scale', '-loss_scale', type=float, default=0,
              help="For FP16 training, the static loss scale to use. If not "
                   "set, the loss scale is dynamically computed.")
    group.add('--apex_opt_level', '-apex_opt_level', type=str, default="O1",
              choices=["O0", "O1", "O2", "O3"],
              help="For FP16 training, the opt_level to use."
                   "See https://nvidia.github.io/apex/amp.html#opt-levels.")


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. "
                   "Options are [text|img|audio|vec].")

    group.add('--train_src', '-train_src', required=True, nargs='+',
              help="Path(s) to the training source data")
    group.add('--train_tgt', '-train_tgt', required=True, nargs='+',
              help="Path(s) to the training target data")
    group.add('--train_align', '-train_align', nargs='+', default=[None],
              help="Path(s) to the training src-tgt alignment")
    group.add('--train_ids', '-train_ids', nargs='+', default=[None],
              help="ids to name training shards, used for corpus weighting")

    group.add('--valid_src', '-valid_src',
              help="Path to the validation source data")
    group.add('--valid_tgt', '-valid_tgt',
              help="Path to the validation target data")
    group.add('--valid_align', '-valid_align', default=None,
              help="Path(s) to the validation src-tgt alignment")

    group.add('--src_dir', '-src_dir', default="",
              help="Source directory for image or audio files.")

    group.add('--save_data', '-save_data', required=True,
              help="Output file for the prepared data")

    group.add('--max_shard_size', '-max_shard_size', type=int, default=0,
              help="""Deprecated use shard_size instead""")

    group.add('--shard_size', '-shard_size', type=int, default=1000000,
              help="Divide src_corpus and tgt_corpus into "
                   "smaller multiple src_copus and tgt corpus files, then "
                   "build shards, each shard will have "
                   "opt.shard_size samples except last shard. "
                   "shard_size=0 means no segmentation "
                   "shard_size>0 means segment dataset into multiple shards, "
                   "each shard has shard_size samples")

    group.add('--num_threads', '-num_threads', type=int, default=1,
              help="Number of shards to build in parallel.")

    group.add('--overwrite', '-overwrite', action="store_true",
              help="Overwrite existing shards if any.")

    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    # if you want to pass an existing vocab.pt file, pass it to
    # -src_vocab alone as it already contains tgt vocab.
    group.add('--src_vocab', '-src_vocab', default="",
              help="Path to an existing source vocabulary. Format: "
                   "one word per line.")
    group.add('--tgt_vocab', '-tgt_vocab', default="",
              help="Path to an existing target vocabulary. Format: "
                   "one word per line.")
    group.add('--features_vocabs_prefix', '-features_vocabs_prefix',
              type=str, default='',
              help="Path prefix to existing features vocabularies")
    group.add('--src_vocab_size', '-src_vocab_size', type=int, default=50000,
              help="Size of the source vocabulary")
    group.add('--tgt_vocab_size', '-tgt_vocab_size', type=int, default=50000,
              help="Size of the target vocabulary")
    group.add('--vocab_size_multiple', '-vocab_size_multiple',
              type=int, default=1,
              help="Make the vocabulary size a multiple of this value")

    group.add('--src_words_min_frequency',
              '-src_words_min_frequency', type=int, default=0)
    group.add('--tgt_words_min_frequency',
              '-tgt_words_min_frequency', type=int, default=0)

    group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
              help="Create dynamic dictionaries")
    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add('--src_seq_length', '-src_seq_length', type=int, default=50,
              help="Maximum source sequence length")
    group.add('--src_seq_length_trunc', '-src_seq_length_trunc',
              type=int, default=None,
              help="Truncate source sequence length.")
    group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=50,
              help="Maximum target sequence length to keep.")
    group.add('--tgt_seq_length_trunc', '-tgt_seq_length_trunc',
              type=int, default=None,
              help="Truncate target sequence length.")
    group.add('--lower', '-lower', action='store_true', help='lowercase data')
    group.add('--filter_valid', '-filter_valid', action='store_true',
              help='Filter validation data by src and/or tgt length')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add('--shuffle', '-shuffle', type=int, default=0,
              help="Shuffle data")
    group.add('--seed', '-seed', type=int, default=3435,
              help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=100000,
              help="Report status every this many sentences")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help="Window size for spectrogram in seconds.")
    group.add('--window_stride', '-window_stride', type=float, default=.01,
              help="Window stride for spectrogram in seconds.")
    group.add('--window', '-window', default='hamming',
              help="Window type for spectrogram generation.")

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3,
              choices=[3, 1],
              help="Using grayscale image can training "
                   "model faster and smaller")


def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add('--data', '-data', required=True,
              help='Path prefix to the ".train.pt" and '
                   '".valid.pt" file path from preprocess.py')

    group.add('--data_ids', '-data_ids', nargs='+', default=[None],
              help="In case there are several corpora.")
    group.add('--data_weights', '-data_weights', type=int, nargs='+',
              default=[1], help="""Weights of different corpora,
              should follow the same order as in -data_ids.""")

    group.add('--save_model', '-save_model', default='model',
              help="Model filename (the model will be saved as "
                   "<save_model>_N.pt where N is the number "
                   "of steps")

    group.add('--save_checkpoint_steps', '-save_checkpoint_steps',
              type=int, default=5000,
              help="""Save a checkpoint every X steps""")
    group.add('--keep_checkpoint', '-keep_checkpoint', type=int, default=-1,
              help="Keep X checkpoints (negative: keep all)")

    # GPU
    group.add('--gpuid', '-gpuid', default=[], nargs='*', type=int,
              help="Deprecated see world_size and gpu_ranks.")
    group.add('--gpu_ranks', '-gpu_ranks', default=[], nargs='*', type=int,
              help="list of ranks of each process.")
    group.add('--world_size', '-world_size', default=1, type=int,
              help="total number of distributed processes.")
    group.add('--gpu_backend', '-gpu_backend',
              default="nccl", type=str,
              help="Type of torch distributed backend")
    group.add('--gpu_verbose_level', '-gpu_verbose_level', default=0, type=int,
              help="Gives more info on each process per GPU.")
    group.add('--master_ip', '-master_ip', default="localhost", type=str,
              help="IP of master for torch.distributed training.")
    group.add('--master_port', '-master_port', default=10000, type=int,
              help="Port of master for torch.distributed training.")
    group.add('--queue_size', '-queue_size', default=400, type=int,
              help="Size of queue for each process in producer/consumer")

    group.add('--seed', '-seed', type=int, default=-1,
              help="Random seed used for the experiments "
                   "reproducibility.")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add('--param_init', '-param_init', type=float, default=0.1,
              help="Parameters are initialized over uniform distribution "
                   "with support (-param_init, param_init). "
                   "Use 0 to not use initialization")
    group.add('--param_init_glorot', '-param_init_glorot', action='store_true',
              help="Init parameters with xavier_uniform. "
                   "Required for transformer.")

    group.add('--train_from', '-train_from', default='', type=str,
              help="If training from a checkpoint then this is the "
                   "path to the pretrained model's state_dict.")
    group.add('--reset_optim', '-reset_optim', default='none',
              choices=['none', 'all', 'states', 'keep_states'],
              help="Optimization resetter when train_from.")

    # Pretrained word vectors
    group.add('--pre_word_vecs_enc', '-pre_word_vecs_enc',
              help="If a valid path is specified, then this will load "
                   "pretrained word embeddings on the encoder side. "
                   "See README for specific formatting instructions.")
    group.add('--pre_word_vecs_dec', '-pre_word_vecs_dec',
              help="If a valid path is specified, then this will load "
                   "pretrained word embeddings on the decoder side. "
                   "See README for specific formatting instructions.")
    # Fixed word vectors
    group.add('--fix_word_vecs_enc', '-fix_word_vecs_enc',
              action='store_true',
              help="Fix word embeddings on the encoder side.")
    group.add('--fix_word_vecs_dec', '-fix_word_vecs_dec',
              action='store_true',
              help="Fix word embeddings on the decoder side.")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add('--batch_size', '-batch_size', type=int, default=64,
              help='Maximum batch size for training')
    group.add('--batch_type', '-batch_type', default='sents',
              choices=["sents", "tokens"],
              help="Batch grouping for batch_size. Standard "
                   "is sents. Tokens will do dynamic batching")
    group.add('--pool_factor', '-pool_factor', type=int, default=8192,
              help="""Factor used in data loading and batch creations.
              It will load the equivalent of `pool_factor` batches,
              sort them by the according `sort_key` to produce
              homogeneous batches and reduce padding, and yield
              the produced batches in a shuffled way.
              Inspired by torchtext's pool mechanism.""")
    group.add('--normalization', '-normalization', default='sents',
              choices=["sents", "tokens"],
              help='Normalization method of the gradient.')
    group.add('--accum_count', '-accum_count', type=int, nargs='+',
              default=[1],
              help="Accumulate gradient this many times. "
                   "Approximately equivalent to updating "
                   "batch_size * accum_count batches at once. "
                   "Recommended for Transformer.")
    group.add('--accum_steps', '-accum_steps', type=int, nargs='+',
              default=[0], help="Steps at which accum_count values change")
    group.add('--valid_steps', '-valid_steps', type=int, default=10000,
              help='Perfom validation every X steps')
    group.add('--valid_batch_size', '-valid_batch_size', type=int, default=32,
              help='Maximum batch size for validation')
    group.add('--max_generator_batches', '-max_generator_batches',
              type=int, default=32,
              help="Maximum batches of words in a sequence to run "
                   "the generator on in parallel. Higher is faster, but "
                   "uses more memory. Set to 0 to disable.")
    group.add('--train_steps', '-train_steps', type=int, default=100000,
              help='Number of training steps')
    group.add('--single_pass', '-single_pass', action='store_true',
              help="Make a single pass over the training dataset.")
    group.add('--epochs', '-epochs', type=int, default=0,
              help='Deprecated epochs see train_steps')
    group.add('--early_stopping', '-early_stopping', type=int, default=0,
              help='Number of validation steps without improving.')
    group.add('--early_stopping_criteria', '-early_stopping_criteria',
              nargs="*", default=None,
              help='Criteria to use for early stopping.')
    group.add('--optim', '-optim', default='sgd',
              choices=['sgd', 'adagrad', 'adadelta', 'adam',
                       'sparseadam', 'adafactor', 'fusedadam'],
              help="Optimization method.")
    group.add('--adagrad_accumulator_init', '-adagrad_accumulator_init',
              type=float, default=0,
              help="Initializes the accumulator values in adagrad. "
                   "Mirrors the initial_accumulator_value option "
                   "in the tensorflow adagrad (use 0.1 for their default).")
    group.add('--max_grad_norm', '-max_grad_norm', type=float, default=5,
              help="If the norm of the gradient vector exceeds this, "
                   "renormalize it to have the norm equal to "
                   "max_grad_norm")
    group.add('--dropout', '-dropout', type=float, default=[0.3], nargs='+',
              help="Dropout probability; applied in LSTM stacks.")
    group.add('--attention_dropout', '-attention_dropout', type=float,
              default=[0.1], nargs='+',
              help="Attention Dropout probability.")
    group.add('--dropout_steps', '-dropout_steps', type=int, nargs='+',
              default=[0], help="Steps at which dropout changes.")
    group.add('--truncated_decoder', '-truncated_decoder', type=int, default=0,
              help="""Truncated bptt.""")
    group.add('--adam_beta1', '-adam_beta1', type=float, default=0.9,
              help="The beta1 parameter used by Adam. "
                   "Almost without exception a value of 0.9 is used in "
                   "the literature, seemingly giving good results, "
                   "so we would discourage changing this value from "
                   "the default without due consideration.")
    group.add('--adam_beta2', '-adam_beta2', type=float, default=0.999,
              help='The beta2 parameter used by Adam. '
                   'Typically a value of 0.999 is recommended, as this is '
                   'the value suggested by the original paper describing '
                   'Adam, and is also the value adopted in other frameworks '
                   'such as Tensorflow and Keras, i.e. see: '
                   'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
                   'Optimizer or https://keras.io/optimizers/ . '
                   'Whereas recently the paper "Attention is All You Need" '
                   'suggested a value of 0.98 for beta2, this parameter may '
                   'not work well for normal models / default '
                   'baselines.')
    group.add('--label_smoothing', '-label_smoothing', type=float, default=0.0,
              help="Label smoothing value epsilon. "
                   "Probabilities of all non-true labels "
                   "will be smoothed by epsilon / (vocab_size - 1). "
                   "Set to zero to turn off label smoothing. "
                   "For more detailed information, see: "
                   "https://arxiv.org/abs/1512.00567")
    group.add('--average_decay', '-average_decay', type=float, default=0,
              help="Moving average decay. "
                   "Set to other than 0 (e.g. 1e-4) to activate. "
                   "Similar to Marian NMT implementation: "
                   "http://www.aclweb.org/anthology/P18-4020 "
                   "For more detail on Exponential Moving Average: "
                   "https://en.wikipedia.org/wiki/Moving_average")
    group.add('--average_every', '-average_every', type=int, default=1,
              help="Step for moving average. "
                   "Default is every update, "
                   "if -average_decay is set.")

    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add('--learning_rate', '-learning_rate', type=float, default=1.0,
              help="Starting learning rate. "
                   "Recommended settings: sgd = 1, adagrad = 0.1, "
                   "adadelta = 1, adam = 0.001")
    group.add('--learning_rate_decay', '-learning_rate_decay',
              type=float, default=0.5,
              help="If update_learning_rate, decay learning rate by "
                   "this much if steps have gone past "
                   "start_decay_steps")
    group.add('--start_decay_steps', '-start_decay_steps',
              type=int, default=50000,
              help="Start decaying every decay_steps after "
                   "start_decay_steps")
    group.add('--decay_steps', '-decay_steps', type=int, default=10000,
              help="Decay every decay_steps")

    group.add('--decay_method', '-decay_method', type=str, default="none",
              choices=['noam', 'noamwd', 'rsqrt', 'none'],
              help="Use a custom decay rate.")
    group.add('--warmup_steps', '-warmup_steps', type=int, default=4000,
              help="Number of warmup steps for custom decay.")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=50,
              help="Print stats at this interval.")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")
    group.add('--exp_host', '-exp_host', type=str, default="",
              help="Send logs to this crayon server.")
    group.add('--exp', '-exp', type=str, default="",
              help="Name of the experiment for logging.")
    # Use Tensorboard for visualization during training
    group.add('--tensorboard', '-tensorboard', action="store_true",
              help="Use tensorboard for visualization during training. "
                   "Must have the library tensorboard >= 1.14.")
    group.add("--tensorboard_log_dir", "-tensorboard_log_dir",
              type=str, default="runs/onmt",
              help="Log directory for Tensorboard. "
                   "This is also the name of the run.")

    group = parser.add_argument_group('Speech')
    # Options most relevant to speech
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help="Window size for spectrogram in seconds.")

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3, choices=[3, 1],
              help="Using grayscale image can training "
                   "model faster and smaller")


def translate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add('--model', '-model', dest='models', metavar='MODEL',
              nargs='+', type=str, default=[], required=True,
              help="Path to model .pt file(s). "
                   "Multiple models can be specified, "
                   "for ensemble decoding.")
    group.add('--fp32', '-fp32', action='store_true',
              help="Force the model to be in FP32 "
                   "because FP16 is very slow on GTX1080(ti).")
    group.add('--avg_raw_probs', '-avg_raw_probs', action='store_true',
              help="If this is set, during ensembling scores from "
                   "different models will be combined by averaging their "
                   "raw probabilities and then taking the log. Otherwise, "
                   "the log probabilities will be averaged directly. "
                   "Necessary for models whose output layers can assign "
                   "zero probability.")

    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. Options: [text|img].")

    group.add('--src', '-src', required=True,
              help="Source sequence to decode (one line per "
                   "sequence)")
    group.add('--src_dir', '-src_dir', default="",
              help='Source directory for image or audio files')
    group.add('--tgt', '-tgt',
              help='True target sequence (optional)')
    group.add('--shard_size', '-shard_size', type=int, default=10000,
              help="Divide src and tgt (if applicable) into "
                   "smaller multiple src and tgt files, then "
                   "build shards, each shard will have "
                   "opt.shard_size samples except last shard. "
                   "shard_size=0 means no segmentation "
                   "shard_size>0 means segment dataset into multiple shards, "
                   "each shard has shard_size samples")
    group.add('--output', '-output', default='pred.txt',
              help="Path to output the predictions (each line will "
                   "be the decoded sequence")
    group.add('--report_align', '-report_align', action='store_true',
              help="Report alignment for each translation.")
    group.add('--report_bleu', '-report_bleu', action='store_true',
              help="Report bleu score after translation, "
                   "call tools/multi-bleu.perl on command line")
    group.add('--report_rouge', '-report_rouge', action='store_true',
              help="Report rouge 1/2/3/L/SU4 score after translation "
                   "call tools/test_rouge.py on command line")
    group.add('--report_time', '-report_time', action='store_true',
              help="Report some translation time metrics")

    # Options most relevant to summarization.
    group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
              help="Create dynamic dictionaries")
    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    group = parser.add_argument_group('Random Sampling')
    group.add('--random_sampling_topk', '-random_sampling_topk',
              default=1, type=int,
              help="Set this to -1 to do random sampling from full "
                   "distribution. Set this to value k>1 to do random "
                   "sampling restricted to the k most likely next tokens. "
                   "Set this to 1 to use argmax or for doing beam "
                   "search.")
    group.add('--random_sampling_temp', '-random_sampling_temp',
              default=1., type=float,
              help="If doing random sampling, divide the logits by "
                   "this before computing softmax during decoding.")
    group.add('--seed', '-seed', type=int, default=829,
              help="Random seed")

    group = parser.add_argument_group('Beam')
    group.add('--beam_size', '-beam_size', type=int, default=5,
              help='Beam size')
    group.add('--min_length', '-min_length', type=int, default=0,
              help='Minimum prediction length')
    group.add('--max_length', '-max_length', type=int, default=100,
              help='Maximum prediction length.')
    group.add('--max_sent_length', '-max_sent_length', action=DeprecateAction,
              help="Deprecated, use `-max_length` instead")

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add('--stepwise_penalty', '-stepwise_penalty', action='store_true',
              help="Apply penalty at every decoding step. "
                   "Helpful for summary penalty.")
    group.add('--length_penalty', '-length_penalty', default='none',
              choices=['none', 'wu', 'avg'],
              help="Length Penalty to use.")
    group.add('--ratio', '-ratio', type=float, default=-0.,
              help="Ratio based beam stop condition")
    group.add('--coverage_penalty', '-coverage_penalty', default='none',
              choices=['none', 'wu', 'summary'],
              help="Coverage Penalty to use.")
    group.add('--alpha', '-alpha', type=float, default=0.,
              help="Google NMT length penalty parameter "
                   "(higher = longer generation)")
    group.add('--beta', '-beta', type=float, default=-0.,
              help="Coverage penalty parameter")
    group.add('--block_ngram_repeat', '-block_ngram_repeat',
              type=int, default=0,
              help='Block repetition of ngrams during decoding.')
    group.add('--ignore_when_blocking', '-ignore_when_blocking',
              nargs='+', type=str, default=[],
              help="Ignore these strings when blocking repeats. "
                   "You want to block sentence delimiters.")
    group.add('--replace_unk', '-replace_unk', action="store_true",
              help="Replace the generated UNK tokens with the "
                   "source token that had highest attention weight. If "
                   "phrase_table is provided, it will look up the "
                   "identified source token and give the corresponding "
                   "target token. If it is not provided (or the identified "
                   "source token does not exist in the table), then it "
                   "will copy the source token.")
    group.add('--phrase_table', '-phrase_table', type=str, default="",
              help="If phrase_table is provided (with replace_unk), it will "
                   "look up the identified source token and give the "
                   "corresponding target token. If it is not provided "
                   "(or the identified source token does not exist in "
                   "the table), then it will copy the source token.")
    group = parser.add_argument_group('Logging')
    group.add('--verbose', '-verbose', action="store_true",
              help='Print scores and predictions for each sentence')
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")
    group.add('--attn_debug', '-attn_debug', action="store_true",
              help='Print best attn for each word')
    group.add('--align_debug', '-align_debug', action="store_true",
              help='Print best align for each word')
    group.add('--dump_beam', '-dump_beam', type=str, default="",
              help='File to dump beam information to.')
    group.add('--n_best', '-n_best', type=int, default=1,
              help="If verbose is set, will output the n_best "
                   "decoded sentences")

    group = parser.add_argument_group('Efficiency')
    group.add('--batch_size', '-batch_size', type=int, default=30,
              help='Batch size')
    group.add('--batch_type', '-batch_type', default='sents',
              choices=["sents", "tokens"],
              help="Batch grouping for batch_size. Standard "
                   "is sents. Tokens will do dynamic batching")
    group.add('--gpu', '-gpu', type=int, default=-1,
              help="Device to run on")

    # Options most relevant to speech.
    group = parser.add_argument_group('Speech')
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help='Window size for spectrogram in seconds')
    group.add('--window_stride', '-window_stride', type=float, default=.01,
              help='Window stride for spectrogram in seconds')
    group.add('--window', '-window', default='hamming',
              help='Window type for spectrogram generation')

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3, choices=[3, 1],
              help="Using grayscale image can training "
                   "model faster and smaller")


# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.


class StoreLoggingLevelAction(configargparse.Action):
    """ Convert string to logging level """
    import logging
    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(StoreLoggingLevelAction, self).__init__(
            option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        # Get the key 'value' in the dict, or just use 'value'
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)


class DeprecateAction(configargparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise configargparse.ArgumentTypeError(msg)
