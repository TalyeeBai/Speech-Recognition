import argparse
import wer
import re


# create a function that calls wer.string_edit_distance() on every utterance
# and accumulates the errors for the corpus. Then, report the word error rate (WER)
# and the sentence error rate (SER). The WER should include the the total errors as well as the
# separately reporting the percentage of insertions, deletions and substitutions.
# The function signature is
# num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(ref=reference_string, hyp=hypothesis_string)
#
def score(ref_trn=None, hyp_trn=None):
    ref_dict = read_trn(ref_trn)
    hyp_dict = read_trn(hyp_trn)
    assert len(ref_dict) == len(hyp_dict)
    word2idx = {}
    word_to_idx(ref_dict, word2idx)
    word_to_idx(hyp_dict, word2idx)

    Err = 0
    Sub = 0
    Del = 0
    Ins = 0
    Tks = 0
    ErrSen = 0
    for hyp_k, hyp_v in hyp_dict.items():
        if hyp_k not in ref_dict:
            raise RuntimeError("key {} not found in ref".format(hyp_k))
        ref_v = ref_dict[hyp_k]
        ref_s = sen_to_idx_str(ref_v, word2idx)
        hyp_s = sen_to_idx_str(hyp_v, word2idx)
        (N, E, D, I, S) = wer.string_edit_distance(ref_s, hyp_s)
        if E > 0:
            ErrSen = ErrSen + 1
        Tks = Tks + N
        Err = Err + E
        Sub = Sub + S
        Del = Del + D
        Ins = Ins + I

        print("id: ({})".format(hyp_k))
        print("Scores: N={}, S={}, D={}, I={}\n".format(N, S, D, I))

    N = len(ref_dict)
    print("-----------------------------------")
    print("Sentence Error Rate:")
    print("Sum: N={}, Err={}".format(N, ErrSen))
    print("Avg: N={}, Err={:0.2f}%".format(N, 100. * ErrSen / N))
    print("-----------------------------------")
    print("Word Error Rate:")
    print("Sum: N={}, Err={}, Sub={}, Del={}, Ins={}".format(Tks, Err, Sub, Del, Ins))
    print("Avg: N={}, Err={:0.2f}%, Sub={:0.2f}%, Del={:0.2f}%, Ins={:0.2f}%".format(
        Tks, 100. * Err / Tks, 100. * Sub / Tks, 100. * Del / Tks, 100. * Ins / Tks
    ))
    print("-----------------------------------")


def sen_to_idx_str(sen, word2idx):
    words = sen.split()
    idxs = []
    for word in words:
        idxs.append(chr(word2idx[word]))
    return "".join(idxs)


def word_to_idx(dict, word2idx):
    for _, v in dict.items():
        words = v.split()
        for word in words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)


def read_trn(fn):
    id2sens = {}
    with open(fn) as fp:
        for line in fp:
            match_obj = re.match(r'(.*) \((.*)\)', line)
            if not match_obj:
                raise RuntimeError("Bad trn file line {}".format(line))

            id2sens[match_obj.group(2)] = match_obj.group(1)
    return id2sens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate ASR results.\n"
                                                 "Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument('-ht', '--hyptrn', help='Hypothesized transcripts in TRN format', required=True, default=None)
    parser.add_argument('-rt', '--reftrn', help='Reference transcripts in TRN format', required=True, default=None)
    args = parser.parse_args()

    if args.reftrn is None or args.hyptrn is None:
        RuntimeError("Must specify reference trn and hypothesis trn files.")

    score(ref_trn=args.reftrn, hyp_trn=args.hyptrn)
