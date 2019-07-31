import csv
from gensim.models import Word2Vec
import os
from argparse import ArgumentParser
import json
from nltk.tokenize import sent_tokenize
from modified_extractor import findSVOs
from extractor.subject_verb_object_extract import nlp
from tools import CONTEXT_LINE

class Word2VecCorpus:
    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
    def __iter__(self):
        with open(self.data_path, encoding="UTF8") as csv_file:
            reader = csv.reader(csv_file, quotechar='"')
            for idx, line in enumerate(reader):
                for sent in sent_tokenize(line[CONTEXT_LINE]):
                    yield self.tokenizer.tokenize(sent)

class MyTokenizer():
    def getSVO(self, sent):
        tokens = nlp(sent)
        svos = findSVOs(tokens)
        return svos

    def tokenize(self, sent):
        tokens = nlp(sent)
        return [token.lemma_ for token in tokens]


class EmbeddingGenerator():
    def __init__(self, data_path, save_path, config):
        self.data_path = data_path
        self.save_path = save_path
        self.config = config
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        max_dir_num = 0
        all_subdir = [int(s) for s in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, str(s)))]
        if all_subdir:
            max_dir_num = max(all_subdir)
        max_dir_num += 1
        self.model_path = os.path.join(save_path, str(max_dir_num))
        os.mkdir(self.model_path)
        self.model_name = "word2vec.model"
        self.config_name = "config.json"

    def generate(self):
        word2vec_corpus = Word2VecCorpus(self.data_path, MyTokenizer())
        word2vec_model = Word2Vec(
            word2vec_corpus,
            size=self.config.size,
            alpha=self.config.alpha,
            window=self.config.window,
            min_count=self.config.min_count,
            sg=self.config.sg,
            negative=self.config.negative)

        word2vec_model.save(os.path.join(self.model_path, self.model_name))

        if hasattr(self.config, "device"):
            device = self.config.device
            self.config.device = "gpu"

        with open(os.path.join(self.model_path, self.config_name), 'w') as outfile:
            json.dump(vars(self.config), outfile)

        if hasattr(self.config, "device"):
            self.config.device = device

        return self.model_path

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--train_path", dest="train_path", default="data/train.csv")
    parser.add_argument("--dict_path", dest="dict_path", default="word2vec")
    parser.add_argument("--size",dest="size", type=int, default=200)
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.025)
    parser.add_argument("--window", dest="window", type=int, default=5)
    parser.add_argument("--min_count", dest="min_count", type=int, default=0)
    parser.add_argument("--sg", dest="sg", type=int, default=0)
    parser.add_argument("--negative", dest="negative", type=int, default=5)
    config = parser.parse_args()
    return config

if __name__ == "__main__":
    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    config = build_parser()
    _print_config(config)

    generator = EmbeddingGenerator(config.train_path, config.dict_path, config)
    generator.generate()

    print("DONE...")
