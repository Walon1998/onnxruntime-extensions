import json
from collections import OrderedDict

from ._base import ProcessingTracedModule, tensor_data_type as _dt
from ._torchext import create_op_function
from ._onnx_ops import schema
from .._ocos import default_opset_domain


def make_custom_op(ctx, op_type, input_names, output_names, container, operator_name=None, **kwargs):
    op_name = container.get_unique_operator_name(op_type) if operator_name is None else operator_name
    container.add_node(op_type, input_names, output_names,
                       op_version=1, name=op_name, op_domain=default_opset_domain(), **kwargs)


def create_bert_tokenizer(ctx, name, input_names, output_names, container, operator_name=None, **kwargs):
    if 'vocab_file' in kwargs:
        attrs = dict(vocab_file=kwargs['vocab_file'])
        for a in ['strip_accents', 'do_lower_case', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token', 'tokenize_chinese_chars', 'max_length', 'do_basic_tokenize', 'suffix_indicator', 'truncation_strategy_name']:
            if a in kwargs and kwargs[a] is not None:
                attrs[a] = kwargs[a]
    else:
        raise RuntimeError("Need vocab_file parameter to build the tokenizer")

    return make_custom_op(ctx, name, input_names,
                          output_names, container, operator_name=operator_name, **attrs)


@schema(inputs=((_dt.STRING, []),),
        outputs=((_dt.INT64, []), (_dt.INT64, []), (_dt.INT64, [])))
def bert_tokenizer(ctx, input_names, output_names, container, operator_name=None, **kwargs):
    return create_bert_tokenizer(ctx, 'BertTokenizer', input_names, output_names,
                                 container, operator_name=operator_name, **kwargs)


@schema(inputs=((_dt.STRING, []),),
        outputs=((_dt.INT64, []), (_dt.INT64, []), (_dt.INT64, [])))
def hf_bert_tokenizer(ctx, input_names, output_names, container, operator_name=None, **kwargs):
    return create_bert_tokenizer(ctx, 'HfBertTokenizer', input_names, output_names,
                                 container, operator_name=operator_name, **kwargs)


@schema(inputs=((_dt.STRING, []),),
        outputs=((_dt.INT64, []), (_dt.INT64, [])))
def gpt2_tokenize(ctx, input_names, output_names, container, operator_name=None, **kwargs):
    if 'hf_tok' in kwargs:
        hf_gpt2_tokenizer = kwargs['hf_tok']
        attrs = {'vocab': json.dumps(hf_gpt2_tokenizer.encoder, separators=(',', ':'))}
        sorted_merges = {v_: k_ for k_, v_ in hf_gpt2_tokenizer.bpe_ranks.items()}
        attrs['merges'] = '\n'.join("{} {}".format(*sorted_merges[n_]) for n_ in range(len(sorted_merges)))
    elif 'vocab' in kwargs:
        attrs = dict(
            vocab=kwargs['vocab'],
            merges=kwargs['merges'])
    else:
        raise RuntimeError("Need hf_tok/vocab parameter to build the tokenizer")
    padding_len = -1
    if 'padding_length' in kwargs:
        padding_len = kwargs['padding_length']
    attrs['padding_length'] = padding_len

    return make_custom_op(ctx, 'GPT2Tokenizer', input_names,
                          output_names, container, operator_name=operator_name, **attrs)


def _get_file_content(path):
    with open(path, "rb") as file:
        return file.read()


def _get_bound_object(func):
    return func.__self__

# v1. Order of outputs - input_ids, token_type_ids, attention_mask
#    (this is NOT consistent with the HuggingFace implementation of the tokenizer)
class PreHuggingFaceBert(ProcessingTracedModule):
    def __init__(self, hf_tok=None, vocab_file=None, do_lower_case=None, strip_accents=None, unk_token=None, sep_token=None, pad_token=None, cls_token=None, mask_token=None, tokenize_chinese_chars=None, max_length=None, do_basic_tokenize=True, suffix_indicator=None, truncation_strategy_name=None):
        super(PreHuggingFaceBert, self).__init__()

        if hf_tok is not None and vocab_file is None:
            ordered_vocab = OrderedDict(sorted(hf_tok.vocab.items(), key=lambda item: int(item[1])))
            vocab_file = '\n'.join(ordered_vocab.keys())

        if hf_tok is not None and do_lower_case is None:
            do_lower_case = 1 if hasattr(hf_tok, 'do_lower_case') and hf_tok.do_lower_case else 0

        if hf_tok is not None and strip_accents is None:
            strip_accents = 1 if 'strip_accents' in hf_tok.init_kwargs and hf_tok.init_kwargs.get('strip_accents') else 0

        if hf_tok is not None and unk_token is None:
            unk_token = hf_tok.special_tokens_map["unk_token"]

        if hf_tok is not None and sep_token is None:
            sep_token = hf_tok.special_tokens_map["sep_token"]

        if hf_tok is not None and pad_token is None:
            pad_token = hf_tok.special_tokens_map["pad_token"]

        if hf_tok is not None and cls_token is None:
            cls_token = hf_tok.special_tokens_map["cls_token"]

        if hf_tok is not None and mask_token is None:
            mask_token = hf_tok.special_tokens_map["mask_token"]

        if hf_tok is not None and max_length is None:
            max_length = hf_tok.model_max_length

        if hf_tok is not None and tokenize_chinese_chars is None:
            tokenize_chinese_chars = 1 if 'tokenize_chinese_chars' in hf_tok.init_kwargs and hf_tok.init_kwargs.get('tokenize_chinese_chars') else 0


        self.onnx_bert_tokenizer = create_op_function('BertTokenizer', bert_tokenizer,
                                                          vocab_file=vocab_file,
                                                          do_lower_case=do_lower_case,
                                                          strip_accents=strip_accents,
                                                          unk_token=unk_token,
                                                          sep_token=sep_token,
                                                          pad_token=pad_token,
                                                          cls_token=cls_token,
                                                          mask_token=mask_token,
                                                          tokenize_chinese_chars=tokenize_chinese_chars,
                                                          max_length=max_length,
                                                          do_basic_tokenize=do_basic_tokenize,
                                                          suffix_indicator=suffix_indicator,
                                                          truncation_strategy_name=truncation_strategy_name,
                                                          )


    def forward(self, text):
        return self.onnx_bert_tokenizer(text)

    def export(self, *args, **kwargs):
        return _get_bound_object(self.onnx_bert_tokenizer).build_model(kwargs.get('opset_version', 0), *args)


# v2. Order of outputs - input_ids, attention_mask, token_type_ids
#    (this is consistent with the HuggingFace implementation of the tokenizer)
class HfBertTokenizer(ProcessingTracedModule):
    def __init__(self, hf_tok=None, vocab_file=None, do_lower_case=0, strip_accents=1):
        super(HfBertTokenizer, self).__init__()
        if hf_tok is None:
            self.onnx_bert_tokenizer = create_op_function('HfBertTokenizer', hf_bert_tokenizer,
                                                          vocab_file=vocab_file,
                                                          do_lower_case=do_lower_case,
                                                          strip_accents=strip_accents)
        else:
            self.onnx_bert_tokenizer = create_op_function('HfBertTokenizer', hf_bert_tokenizer,
                                                          hf_tok=hf_tok)

    def forward(self, text):
        return self.onnx_bert_tokenizer(text)

    def export(self, *args, **kwargs):
        return _get_bound_object(self.onnx_bert_tokenizer).build_model(kwargs.get('opset_version', 0), *args)


class PreHuggingFaceGPT2(ProcessingTracedModule):
    def __init__(self, hf_tok=None, vocab_file=None, merges_file=None, padding_length=-1):
        super(PreHuggingFaceGPT2, self).__init__()
        if hf_tok is None:
            self.onnx_gpt2_tokenize = create_op_function('GPT2Tokenizer', gpt2_tokenize,
                                                         vocab=_get_file_content(vocab_file),
                                                         merges=_get_file_content(merges_file),
                                                         padding_length=padding_length)
        else:
            self.onnx_gpt2_tokenize = create_op_function('GPT2Tokenizer', gpt2_tokenize, hf_tok=hf_tok)

    def forward(self, text):
        return self.onnx_gpt2_tokenize(text)

    def export(self, *args, **kwargs):
        return _get_bound_object(self.onnx_gpt2_tokenize).build_model(kwargs.get('opset_version', 0), *args)
