# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

from pygtrie import CharTrie
import sys

# ============================[ Local Imports ]==========================

from . import corpus


# =====================[ Baseline Extrapolator Model ]===================

class DSFts5BaselineCompleter:

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, training_corpus, completions=6):
        assert isinstance(training_corpus, corpus.DSCorpus)
        self.trie = CharTrie()
        self.min_num_completions = completions
        total = sum(len(tokens) for _, tokens in training_corpus.data.items())
        done = 0
        print("Building FTS-5 trie ...")
        for token in (token for _, tokens in training_corpus.data.items() for token in tokens):
            done += 1
            self.print_progress(done, total)
            parts = token.string.split()
            for part in parts:
                if part in self.trie:
                    self.trie[part] += 1
                else:
                    self.trie[part] = 1
        print("")

    def extrapolate(self, embedding_featureset, prefix_chars, prefix_classes, *unused):
        """
        Use this method to predict ranked postfixes for the given prefix with this model.
        """
        completions = []
        prefix_chars = prefix_chars.split()[-1]
        if len(prefix_chars) > 2:
            completions = sorted(
                self.trie.items(prefix_chars),
                key=lambda completion: completion[1],
                reverse=True)[:self.min_num_completions]
            completions = [(
                    completion[0][len(prefix_chars):],
                    prefix_classes[-1]*(len(completion[0])-len(prefix_chars)),
                    completion[1]
                )
                for completion in completions]
        # Make sure that there are at least as many completions as requested
        if len(completions) < self.min_num_completions:
            completions += [('', [], 0)]*(len(completions) - self.min_num_completions)
        return completions

    @staticmethod
    def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=10):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

