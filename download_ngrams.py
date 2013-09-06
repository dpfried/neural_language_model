from nltk import FreqDist
import subprocess
import io
import gzip
import cPickle
import numpy as np
import h5py
from collections import defaultdict
from corpus_creator import DTYPE

base_url_1grams='http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-1gram-20120701-%s.gz'
base_url_5grams='http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-5gram-20120701-%s.gz'

# urls_1grams = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'other', 'p', 'pos', 'punctuation', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',]
urls_1grams = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'other', 'p', 'pos', 'punctuation', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',]
urls_5grams = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_ADJ_', '_ADP_', '_ADV_', '_CONJ_', '_DET_', '_NOUN_', '_NUM_', '_PRON_', '_PRT_', '_VERB_', 'a_', 'aa', 'ab', 'ac', 'ad', 'ae', 'af', 'ag', 'ah', 'ai', 'aj', 'ak', 'al', 'am', 'an', 'ao', 'ap', 'aq', 'ar', 'as', 'at', 'au', 'av', 'aw', 'ax', 'ay', 'az', 'b_', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'bg', 'bh', 'bi', 'bj', 'bk', 'bl', 'bm', 'bn', 'bo', 'bp', 'bq', 'br', 'bs', 'bt', 'bu', 'bv', 'bw', 'bx', 'by', 'bz', 'c_', 'ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cg', 'ch', 'ci', 'cj', 'ck', 'cl', 'cm', 'cn', 'co', 'cp', 'cq', 'cr', 'cs', 'ct', 'cu', 'cv', 'cw', 'cx', 'cy', 'cz', 'd_', 'da', 'db', 'dc', 'dd', 'de', 'df', 'dg', 'dh', 'di', 'dj', 'dk', 'dl', 'dm', 'dn', 'do', 'dp', 'dq', 'dr', 'ds', 'dt', 'du', 'dv', 'dw', 'dx', 'dy', 'dz', 'e_', 'ea', 'eb', 'ec', 'ed', 'ee', 'ef', 'eg', 'eh', 'ei', 'ej', 'ek', 'el', 'em', 'en', 'eo', 'ep', 'eq', 'er', 'es', 'et', 'eu', 'ev', 'ew', 'ex', 'ey', 'ez', 'f_', 'fa', 'fb', 'fc', 'fd', 'fe', 'ff', 'fg', 'fh', 'fi', 'fj', 'fk', 'fl', 'fm', 'fn', 'fo', 'fp', 'fq', 'fr', 'fs', 'ft', 'fu', 'fv', 'fw', 'fx', 'fy', 'fz', 'g_', 'ga', 'gb', 'gc', 'gd', 'ge', 'gf', 'gg', 'gh', 'gi', 'gj', 'gk', 'gl', 'gm', 'gn', 'go', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gv', 'gw', 'gx', 'gy', 'gz', 'h_', 'ha', 'hb', 'hc', 'hd', 'he', 'hf', 'hg', 'hh', 'hi', 'hj', 'hk', 'hl', 'hm', 'hn', 'ho', 'hp', 'hq', 'hr', 'hs', 'ht', 'hu', 'hv', 'hw', 'hx', 'hy', 'hz', 'i_', 'ia', 'ib', 'ic', 'id', 'ie', 'if', 'ig', 'ih', 'ii', 'ij', 'ik', 'il', 'im', 'in', 'io', 'ip', 'iq', 'ir', 'is', 'it', 'iu', 'iv', 'iw', 'ix', 'iy', 'iz', 'j_', 'ja', 'jb', 'jc', 'jd', 'je', 'jf', 'jg', 'jh', 'ji', 'jj', 'jk', 'jl', 'jm', 'jn', 'jo', 'jp', 'jq', 'jr', 'js', 'jt', 'ju', 'jv', 'jw', 'jx', 'jy', 'jz', 'k_', 'ka', 'kb', 'kc', 'kd', 'ke', 'kf', 'kg', 'kh', 'ki', 'kj', 'kk', 'kl', 'km', 'kn', 'ko', 'kp', 'kq', 'kr', 'ks', 'kt', 'ku', 'kv', 'kw', 'kx', 'ky', 'kz', 'l_', 'la', 'lb', 'lc', 'ld', 'le', 'lf', 'lg', 'lh', 'li', 'lj', 'lk', 'll', 'lm', 'ln', 'lo', 'lp', 'lq', 'lr', 'ls', 'lt', 'lu', 'lv', 'lw', 'lx', 'ly', 'lz', 'm_', 'ma', 'mb', 'mc', 'md', 'me', 'mf', 'mg', 'mh', 'mi', 'mj', 'mk', 'ml', 'mm', 'mn', 'mo', 'mp', 'mq', 'mr', 'ms', 'mt', 'mu', 'mv', 'mw', 'mx', 'my', 'mz', 'n_', 'na', 'nb', 'nc', 'nd', 'ne', 'nf', 'ng', 'nh', 'ni', 'nj', 'nk', 'nl', 'nm', 'nn', 'no', 'np', 'nq', 'nr', 'ns', 'nt', 'nu', 'nv', 'nw', 'nx', 'ny', 'nz', 'o_', 'oa', 'ob', 'oc', 'od', 'oe', 'of', 'og', 'oh', 'oi', 'oj', 'ok', 'ol', 'om', 'on', 'oo', 'op', 'oq', 'or', 'os', 'ot', 'other', 'ou', 'ov', 'ow', 'ox', 'oy', 'oz', 'p_', 'pa', 'pb', 'pc', 'pd', 'pe', 'pf', 'pg', 'ph', 'pi', 'pj', 'pk', 'pl', 'pm', 'pn', 'po', 'pp', 'pq', 'pr', 'ps', 'pt', 'pu', 'punctuation', 'pv', 'pw', 'px', 'py', 'pz', 'q_', 'qa', 'qb', 'qc', 'qd', 'qe', 'qf', 'qg', 'qh', 'qi', 'qj', 'ql', 'qm', 'qn', 'qo', 'qp', 'qq', 'qr', 'qs', 'qt', 'qu', 'qv', 'qw', 'qx', 'qy', 'qz', 'r_', 'ra', 'rb', 'rc', 'rd', 're', 'rf', 'rg', 'rh', 'ri', 'rj', 'rk', 'rl', 'rm', 'rn', 'ro', 'rp', 'rq', 'rr', 'rs', 'rt', 'ru', 'rv', 'rw', 'rx', 'ry', 'rz', 's_', 'sa', 'sb', 'sc', 'sd', 'se', 'sf', 'sg', 'sh', 'si', 'sj', 'sk', 'sl', 'sm', 'sn', 'so', 'sp', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw', 'sx', 'sy', 'sz', 't_', 'ta', 'tb', 'tc', 'td', 'te', 'tf', 'tg', 'th', 'ti', 'tj', 'tk', 'tl', 'tm', 'tn', 'to', 'tp', 'tq', 'tr', 'ts', 'tt', 'tu', 'tv', 'tw', 'tx', 'ty', 'tz', 'u_', 'ua', 'ub', 'uc', 'ud', 'ue', 'uf', 'ug', 'uh', 'ui', 'uj', 'uk', 'ul', 'um', 'un', 'uo', 'up', 'uq', 'ur', 'us', 'ut', 'uu', 'uv', 'uw', 'ux', 'uy', 'uz', 'v_', 'va', 'vb', 'vc', 'vd', 've', 'vf', 'vg', 'vh', 'vi', 'vj', 'vk', 'vl', 'vm', 'vn', 'vo', 'vp', 'vq', 'vr', 'vs', 'vt', 'vu', 'vv', 'vw', 'vx', 'vy', 'vz', 'w_', 'wa', 'wb', 'wc', 'wd', 'we', 'wf', 'wg', 'wh', 'wi', 'wj', 'wk', 'wl', 'wm', 'wn', 'wo', 'wp', 'wq', 'wr', 'ws', 'wt', 'wu', 'wv', 'ww', 'wx', 'wy', 'wz', 'x_', 'xa', 'xb', 'xc', 'xd', 'xe', 'xf', 'xg', 'xh', 'xi', 'xj', 'xk', 'xl', 'xm', 'xn', 'xo', 'xp', 'xq', 'xr', 'xs', 'xt', 'xu', 'xv', 'xw', 'xx', 'xy', 'xz', 'y_', 'ya', 'yb', 'yc', 'yd', 'ye', 'yf', 'yg', 'yh', 'yi', 'yj', 'yk', 'yl', 'ym', 'yn', 'yo', 'yp', 'yq', 'yr', 'ys', 'yt', 'yu', 'yv', 'yw', 'yx', 'yy', 'yz', 'z_', 'za', 'zb', 'zc', 'zd', 'ze', 'zf', 'zg', 'zh', 'zi', 'zj', 'zk', 'zl', 'zm', 'zn', 'zo', 'zp', 'zq', 'zr', 'zs', 'zt', 'zu', 'zv', 'zw', 'zx', 'zy', 'zz',]

def buffered_download(url):
    curl_proc = subprocess.Popen(["curl", url], stdout=subprocess.PIPE)
    gzip_proc = subprocess.Popen(["gunzip", "-"], stdin=curl_proc.stdout, stdout=subprocess.PIPE)
    for line in io.open(gzip_proc.stdout.fileno()):
        yield line.rstrip('\n')

def parse_line(line):
    token_string, year_string, total_count_string, book_count_string = line.split('\t')
    tokens = token_string.split()
    return tokens, int(year_string), int(total_count_string), int(book_count_string)

def save_state(save_state_file, counter, current_line_num, current_url):
    with gzip.open(save_state_file, 'w') as f:
        cPickle.dump({
            'counter': counter,
            'current_line_num': current_line_num,
            'current_url': current_url
        }, f)

def build_vocabulary(save_state_file='state.pkl.gz'):
    counter = FreqDist()
    total_line_count = 0
    for url_suffix in urls_1grams:
        print url_suffix
        current_line_num = 0
        for line in buffered_download(base_url_1grams % url_suffix):
            current_line_num += 1
            total_line_count += 1
            try:
                tokens, year, total_count, _ = parse_line(line)
                counter.inc(tokens[0], total_count)
            except:
                print "error parsing line"
                print line
        if save_state_file:
            print 'saving state'
            save_state(save_state_file, counter, current_line_num, url_suffix)
    return counter

def initialize_hd5_file(hd5_filename, frequency_counts=None):
    if frequency_counts is None:
        print '...first pass: building frequency counts and word to token map for files'
        frequency_counts = build_vocabulary()
    print '...creating hd5 file'

    hd5_file = h5py.File(hd5_filename, 'w')

    words, frequencies = zip(*[(word.encode('ascii', 'ignore'), count) for (word, count) in frequency_counts.items()
                               if '_' in word])

    word_array = np.array(['RARE'] + list(words))
    frequency_array = np.array([0] + list(frequencies))

    hd5_file.create_dataset("words", data=word_array)
    hd5_file.create_dataset("word_frequencies", data=frequency_array)

    hd5_file.close()

def add_ngrams_to_hd5(hd5_filename, save_state_file='ngram_state.txt', ngram_length=5, row_chunksize=10000):
    hd5_file = h5py.File(hd5_filename, 'r+')
    print 'building id_map'
    id_map = defaultdict(int, dict(
        (word, index) for (index, word) in enumerate(hd5_file['words'][...])
    ))
    print 'finished building id_map'

    n_rows = 1
    cols = ngram_length + 3 # for year, total count, and book count

    ngram_dset = hd5_file.create_dataset("%d_grams" % ngram_length,
                                   shape=(n_rows, cols),
                                   dtype=DTYPE,
                                   maxshape=(None, cols),
                                   chunks=(row_chunksize, cols))
    total_n_rows = 0
    print 'parsing streams'
    for url_suffix in urls_5grams:
        print url_suffix
        vectors = []
        vector_count = 0
        for line in buffered_download(base_url_5grams % url_suffix):
            tokens, year, total_count, book_count = parse_line(line)
            if all('_' in token for token in tokens):
                sym_ids = [id_map[token] for token in tokens]
                vectors.append(np.array(sym_ids + [year, total_count, book_count]))
                vector_count += 1
                if vector_count % row_chunksize == 0:
                    total_n_rows += vector_count
                    ngram_dset.resize(total_n_rows, axis=0)
                    ngram_dset[-vector_count:] = np.array(vectors)
                    vectors = []
                    vector_count = 0
        print "writing to hd5 file"
        hd5_file.flush()
        with open(save_state_file, 'a') as f:
            f.write('%s\n' % url_suffix)

if __name__ == "__main__":
    filename = '/cl/nldata/books_google_ngrams_eng/pos_5grams.hd5'
    # initialize_hd5_file(filename)
    add_ngrams_to_hd5(filename)
