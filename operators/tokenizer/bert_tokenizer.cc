#include "bert_tokenizer.hpp"

#include <utility>
#include <iostream>
#include <optional>
#include <list>

BertTokenizerVocab::BertTokenizerVocab(std::string_view vocab) : raw_vocab_(vocab) {
  auto tokens = SplitString(raw_vocab_, "\r\n", true);

  for (size_t i = 0; i < tokens.size(); i++) {
    (vocab_)[tokens[i]] = static_cast<int32_t>(i);
  }
}

bool BertTokenizerVocab::FindToken(const ustring& token) {
  auto utf8_token = std::string(token);

  return vocab_.find(utf8_token) != vocab_.end();
}

bool BertTokenizerVocab::FindTokenId(const ustring& token, int32_t& token_id) {
  auto utf8_token = std::string(token);

  auto it = vocab_.find(utf8_token);
  if (it == vocab_.end()) {
    return false;
  }

  token_id = it->second;
  return true;
}

int32_t BertTokenizerVocab::FindTokenId(const ustring& token) {
  auto utf8_token = std::string(token);

  auto it = vocab_.find(utf8_token);
  if (it == vocab_.end()) {
    ORTX_CXX_API_THROW("[BertTokenizerVocab]: can not find tokens: " + std::string(token), ORT_RUNTIME_EXCEPTION);
  }

  return it->second;
}

WordpieceTokenizer::WordpieceTokenizer(
    std::shared_ptr<BertTokenizerVocab> vocab,
    ustring unk_token,
    ustring suffix_indicator,
    int max_input_chars_per_word) : max_input_chars_per_word_(max_input_chars_per_word),
                                    suffix_indicator_(std::move(suffix_indicator)),
                                    unk_token_(std::move(unk_token)),
                                    vocab_(std::move(vocab)) {
  unk_token_id_ = vocab_->FindTokenId(unk_token_);
}

std::vector<ustring> WordpieceTokenizer::Tokenize(const ustring& text, std::list<OffsetMappingType>& offset_map, bool compute_offset_mapping) {
  std::vector<ustring> result;
  ustring token;
  for (auto c : text) {
    if (c == U' ' && !token.empty()) {
      GreedySearch(token, result);
      token.clear();
      continue;
    }

    token.push_back(c);
  }

  if (!token.empty()) {
    GreedySearch(token, result);
  }

  return result;
}

std::vector<ustring> WordpieceTokenizer::Tokenize(const std::vector<ustring>& tokens, std::list<OffsetMappingType>& offset_map, bool compute_offset_mapping) {
  std::vector<ustring> result;
  for (const auto& token : tokens) {
    GreedySearch(token, result);
  }

  if (compute_offset_mapping) {
    size_t offset = 0;
    OffsetMappingType offset_mapping;

    // Add offset mapping for BOS token
    offset_mapping.push_back(std::make_pair(0, 0));

    for (auto i : result) {
      // Handle special cases for offset mapping
      size_t idx = 0;
      if (idx < std::string(i).size() && std::string(i).at(idx) == '#') {
        while (idx < std::string(i).size() && std::string(i).at(idx) == '#') {
          idx++;
        }
        offset--;
        offset_mapping.emplace_back(std::make_pair(offset, offset + std::string(i).size() - idx));
        offset += (std::string(i).size() - idx) + 1;
      } else if (std::string(i).compare("[UNK]") == 0) {
        offset_mapping.emplace_back(std::make_pair(offset, offset + 1));
        offset += 2;
      } else {
        offset_mapping.emplace_back(std::make_pair(offset, offset + std::string(i).size()));
        offset += std::string(i).size() + 1;
      }
    }
    // Add offset mapping for EOS token
    offset_mapping.emplace_back(std::make_pair(0, 0));

    // Add offset mappings for input in this instance to list of offset mappings for all inputs
    offset_map.emplace_back(offset_mapping);
  }

  return result;
}

std::vector<int64_t> WordpieceTokenizer::Encode(const std::vector<ustring>& tokens) {
  std::vector<int64_t> ids;
  for (const auto& token : tokens) {
    int32_t token_id = -1;
    if (!vocab_->FindTokenId(token, token_id)) {
      ids.push_back(unk_token_id_);
      continue;
    }

    ids.push_back(token_id);
  }
  return ids;
}

void WordpieceTokenizer::GreedySearch(const ustring& token, std::vector<ustring>& tokenized_result) {
  if (static_cast<int64_t>(token.size()) > max_input_chars_per_word_) {
    tokenized_result.push_back(unk_token_);
    return;
  }

  size_t start = 0;
  size_t end = 0;
  ustring substr;
  for (; start < token.size();) {
    end = token.size();
    bool is_found = false;
    // try to found the longest matched sub-token in vocab
    for (; start < end;) {
      substr = static_cast<const ustring>(token.substr(start, end - start));
      if (start > 0) {
        substr = static_cast<const ustring>(suffix_indicator_ + substr);
      }
      if (vocab_->FindToken(substr)) {
        is_found = true;
        break;
      }
      end -= 1;
    }
    // token not found in vocab
    if (!is_found) {
      tokenized_result.push_back(unk_token_);
      break;
    }

    tokenized_result.push_back(substr);
    start = end;
  }
}

void TruncateStrategy::Truncate(std::vector<int64_t>& ids, int32_t max_len) {
  if ((max_len > 0) && (static_cast<size_t>(max_len) < ids.size())) {
    ids.resize(max_len);
  }
}

void TruncateStrategy::Truncate(std::vector<int64_t>& ids1, std::vector<int64_t>& ids2, int32_t max_len) {
  if (max_len < 0 || (ids1.size() + ids2.size() <= static_cast<size_t>(max_len))) {
    return;
  }

  auto ids1_keep_len = ids1.size();
  auto ids2_keep_len = ids2.size();
  auto half_max_len = max_len / 2;

  switch (strategy_) {
    case TruncateStrategyType::LONGEST_FIRST:
    case TruncateStrategyType::LONGEST_FROM_BACK:

      if ((ids1_keep_len > static_cast<size_t>(half_max_len)) && (ids2_keep_len > static_cast<size_t>(half_max_len))) {
        ids1_keep_len = static_cast<size_t>(max_len) - half_max_len;
        ids2_keep_len = half_max_len;
      } else if (ids2_keep_len > ids1_keep_len) {
        ids2_keep_len = static_cast<size_t>(max_len) - ids1_keep_len;
      } else {
        ids1_keep_len = static_cast<size_t>(max_len) - ids2_keep_len;
      }

      if (strategy_ == TruncateStrategyType::LONGEST_FIRST) {
        ids1.resize(ids1_keep_len);
        ids2.resize(ids2_keep_len);
      } else {
        ids1.erase(ids1.begin(), ids1.end() - ids1_keep_len);
        ids2.erase(ids2.begin(), ids2.end() - ids2_keep_len);
      }

      return;
    case TruncateStrategyType::ONLY_FIRST:
      return;
    case TruncateStrategyType::ONLY_SECOND:
      return;
    default:
      return;
  }
}

BertTokenizer::BertTokenizer(
    const std::string& vocab,
    bool do_lower_case,
    bool do_basic_tokenize,
    ustring unk_token,
    ustring sep_token,
    ustring pad_token,
    ustring cls_token,
    ustring mask_token,
    bool tokenize_chinese_chars,
    bool strip_accents,
    ustring suffix_indicator,
    int32_t max_len,
    const std::string& truncation_strategy) : max_length_(max_len),
                                              do_basic_tokenize_(do_basic_tokenize),
                                              truncate_(std::make_unique<TruncateStrategy>(truncation_strategy)) {
  vocab_ = std::make_shared<BertTokenizerVocab>(vocab);

  if (do_basic_tokenize) {
    basic_tokenizer_ = std::make_unique<BasicTokenizer>(
        do_lower_case, tokenize_chinese_chars, strip_accents, true, true);
  }
  wordpiece_tokenizer_ = std::make_unique<WordpieceTokenizer>(
      vocab_, unk_token, suffix_indicator);

  unk_token_id_ = vocab_->FindTokenId(unk_token);
  sep_token_id_ = vocab_->FindTokenId(sep_token);
  pad_token_id_ = vocab_->FindTokenId(pad_token);
  cls_token_id_ = vocab_->FindTokenId(cls_token);
  mask_token_id_ = vocab_->FindTokenId(mask_token);
}

std::vector<ustring> BertTokenizer::Tokenize(const ustring& text, std::list<OffsetMappingType>& offset_map, bool compute_offset_mapping) {
  if (do_basic_tokenize_) {
    return wordpiece_tokenizer_->Tokenize(basic_tokenizer_->Tokenize(text), offset_map, compute_offset_mapping);
  }
  return wordpiece_tokenizer_->Tokenize(text, offset_map, compute_offset_mapping);
}

std::vector<int64_t> BertTokenizer::Encode(const std::vector<ustring>& tokens) {
  return wordpiece_tokenizer_->Encode(tokens);
}

void BertTokenizer::Truncate(std::vector<int64_t>& ids) {
  truncate_->Truncate(ids, (max_length_ > 0 && max_length_ <= 2) ? 0 : max_length_ - 2);
}

void BertTokenizer::Truncate(std::vector<int64_t>& ids1, std::vector<int64_t>& ids2) {
  truncate_->Truncate(ids1, ids2, (max_length_ > 0 && max_length_ <= 3) ? 0 : max_length_ - 3);
}

std::vector<int64_t> BertTokenizer::AddSpecialToken(const std::vector<int64_t>& ids) {
  std::vector<int64_t> result;
  result.reserve(ids.size() + 2);
  result.push_back(cls_token_id_);
  result.insert(result.end(), ids.begin(), ids.end());
  result.push_back(sep_token_id_);
  return result;
}

std::vector<int64_t> BertTokenizer::AddSpecialToken(const std::vector<int64_t>& ids1, const std::vector<int64_t>& ids2) {
  std::vector<int64_t> result;
  result.reserve(ids1.size() + ids2.size() + 3);
  result.push_back(cls_token_id_);
  result.insert(result.end(), ids1.begin(), ids1.end());
  result.push_back(sep_token_id_);
  result.insert(result.end(), ids2.begin(), ids2.end());
  result.push_back(sep_token_id_);
  return result;
}

std::vector<int64_t> BertTokenizer::GenerateTypeId(const std::vector<int64_t>& ids) {
  return std::vector<int64_t>(ids.size() + 2, 0);
}

std::vector<int64_t> BertTokenizer::GenerateTypeId(const std::vector<int64_t>& ids1, const std::vector<int64_t>& ids2) {
  std::vector<int64_t> result;
  result.reserve(ids1.size() + ids2.size() + 3);
  result.insert(result.end(), ids1.size() + 2, 0);
  result.insert(result.end(), ids2.size() + 1, 1);
  return result;
}

TruncateStrategy::TruncateStrategy(std::string_view strategy_name) : strategy_(TruncateStrategyType::LONGEST_FIRST) {
  if (strategy_name == "longest_first") {
    strategy_ = TruncateStrategyType::LONGEST_FIRST;
  } else if (strategy_name == "only_first") {
    strategy_ = TruncateStrategyType::ONLY_FIRST;
  } else if (strategy_name == "only_second") {
    strategy_ = TruncateStrategyType::ONLY_SECOND;
  } else if (strategy_name == "longest_from_back") {
    strategy_ = TruncateStrategyType::LONGEST_FROM_BACK;
  }
}

KernelBertTokenizer::KernelBertTokenizer(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  std::string vocab = ort_.KernelInfoGetAttribute<std::string>(&info, "vocab_file");
  bool do_lower_case = TryToGetAttributeWithDefault("do_lower_case", true);
  bool do_basic_tokenize = TryToGetAttributeWithDefault("do_basic_tokenize", true);
  std::string unk_token = TryToGetAttributeWithDefault("unk_token", std::string("[UNK]"));
  std::string sep_token = TryToGetAttributeWithDefault("sep_token", std::string("[SEP]"));
  std::string pad_token = TryToGetAttributeWithDefault("pad_token", std::string("[PAD]"));
  std::string cls_token = TryToGetAttributeWithDefault("cls_token", std::string("[CLS]"));
  std::string mask_token = TryToGetAttributeWithDefault("mask_token", std::string("[MASK]"));
  bool tokenize_chinese_chars = TryToGetAttributeWithDefault("tokenize_chinese_chars", true);
  bool strip_accents = TryToGetAttributeWithDefault("strip_accents", false);
  std::string suffix_indicator = TryToGetAttributeWithDefault("suffix_indicator", std::string("##"));
  std::string truncation_strategy_name = TryToGetAttributeWithDefault("truncation_strategy_name",
                                                                      std::string("longest_first"));
  int32_t max_len = static_cast<int32_t>(TryToGetAttributeWithDefault("max_length", int64_t(-1)));

  tokenizer_ = std::make_unique<BertTokenizer>(
      vocab, do_lower_case, do_basic_tokenize, ustring(unk_token),
      ustring(sep_token), ustring(pad_token), ustring(cls_token),
      ustring(mask_token), tokenize_chinese_chars, strip_accents,
      ustring(suffix_indicator), max_len, truncation_strategy_name);
}

void KernelBertTokenizer::Compute(const ortc::Tensor<std::string>& input,
                                  ortc::Tensor<int64_t>& output,
                                  ortc::Tensor<int64_t>& output1,
                                  ortc::Tensor<int64_t>& output2,
                                  std::optional<ortc::Tensor<int64_t>*> offset_mapping) const {
  // Setup inputs
  auto& input_data = input.Data();

  if (input_data.size() != 1 && input_data.size() != 2) {
    ORTX_CXX_API_THROW("[BertTokenizer]: only support one or two query.", ORT_INVALID_GRAPH);
  }
  std::vector<int64_t> input_ids;
  std::vector<int64_t> token_type_ids;
  std::list<OffsetMappingType> offset_map;

  // Only compute offset mapping if optional output for it exists.
  bool compute_offset_mapping = false;
  if (offset_mapping.has_value()) {
    compute_offset_mapping = true;
  }

  if (input_data.size() == 1) {
    std::vector<ustring> tokens = tokenizer_->Tokenize(ustring(input_data[0]), offset_map, compute_offset_mapping);
    std::vector<int64_t> encoded = tokenizer_->Encode(tokens);
    tokenizer_->Truncate(encoded);
    input_ids = tokenizer_->AddSpecialToken(encoded);
    token_type_ids = tokenizer_->GenerateTypeId(encoded);
  } else {
    std::vector<ustring> tokens1 = tokenizer_->Tokenize(ustring(input_data[0]), offset_map, compute_offset_mapping);
    std::vector<ustring> tokens2 = tokenizer_->Tokenize(ustring(input_data[1]), offset_map, compute_offset_mapping);
    std::vector<int64_t> encoded1 = tokenizer_->Encode(tokens1);
    std::vector<int64_t> encoded2 = tokenizer_->Encode(tokens2);
    input_ids = tokenizer_->AddSpecialToken(encoded1, encoded2);
    token_type_ids = tokenizer_->GenerateTypeId(encoded1, encoded2);
  }

  std::vector<int64_t> attention_mask(input_ids.size(), 1);

  std::vector<int64_t> output_dim{static_cast<int64_t>(input_ids.size())};

  auto* p_out = output.Allocate(output_dim);
  std::copy(input_ids.begin(), input_ids.end(), p_out);
  auto* p_out1 = output1.Allocate(output_dim);
  std::copy(token_type_ids.begin(), token_type_ids.end(), p_out1);
  auto* p_out2 = output2.Allocate(output_dim);
  std::copy(attention_mask.begin(), attention_mask.end(), p_out2);

  std::vector<int64_t> offset_dim{static_cast<int64_t>(input_ids.size()), 2};  // tuple of offsets for each input id

  if (offset_mapping.has_value()) {
    auto* offset = (*offset_mapping)->Allocate(offset_dim);
    int idx2 = 0;
    for (auto& res : offset_map) {
      for (auto& mapping : res) {
        offset[idx2] = mapping.first;
        idx2++;
        offset[idx2] = mapping.second;
        idx2++;
      }
    }
  }
}

KernelHfBertTokenizer::KernelHfBertTokenizer(const OrtApi& api, const OrtKernelInfo& info)
    : KernelBertTokenizer(api, info) {}

void KernelHfBertTokenizer::Compute(const ortc::Tensor<std::string>& input,
                                    ortc::Tensor<int64_t>& output,
                                    ortc::Tensor<int64_t>& output1,
                                    ortc::Tensor<int64_t>& output2,
                                    std::optional<ortc::Tensor<int64_t>*> offset_mapping) const {
  // Setup inputs
  auto& input_data = input.Data();

  if (input_data.size() != 2) {
    ORTX_CXX_API_THROW("[HfBertTokenizer]: Support only two input strings.", ORT_INVALID_GRAPH);
  }

  std::list<OffsetMappingType> offset_map;

  // Only compute offset mapping if optional output for it exists.
  bool compute_offset_mapping = false;
  if (offset_mapping.has_value()) {
    compute_offset_mapping = true;
  }

  std::vector<ustring> tokens1 = tokenizer_->Tokenize(ustring(input_data[0]), offset_map, compute_offset_mapping);
  std::vector<ustring> tokens2 = tokenizer_->Tokenize(ustring(input_data[1]), offset_map, compute_offset_mapping);
  std::vector<int64_t> encoded1 = tokenizer_->Encode(tokens1);
  std::vector<int64_t> encoded2 = tokenizer_->Encode(tokens2);
  std::vector<int64_t> input_ids = tokenizer_->AddSpecialToken(encoded1, encoded2);
  std::vector<int64_t> token_type_ids = tokenizer_->GenerateTypeId(encoded1, encoded2);
  std::vector<int64_t> attention_mask(input_ids.size(), 1LL);

  const std::vector<int64_t> outer_dims{1LL, static_cast<int64_t>(input_ids.size())};

  auto* p_out = output.Allocate(outer_dims);
  std::copy(input_ids.begin(), input_ids.end(), p_out);
  auto* p_out1 = output1.Allocate(outer_dims);
  std::copy(attention_mask.begin(), attention_mask.end(), p_out1);
  auto* p_out2 = output2.Allocate(outer_dims);
  std::copy(token_type_ids.begin(), token_type_ids.end(), p_out2);

  std::vector<int64_t> offset_dim{static_cast<int64_t>(input_ids.size()), 2};  // tuple of offsets for each input id

  if (offset_mapping.has_value()) {
    auto* offset = (*offset_mapping)->Allocate(offset_dim);
    int idx2 = 0;
    for (auto& res : offset_map) {
      for (auto& mapping : res) {
        offset[idx2] = mapping.first;
        idx2++;
        offset[idx2] = mapping.second;
        idx2++;
      }
    }
  }
}
