from collections import OrderedDict
import numpy as np


CODON_TO_AMINO_ACID = {
    'TCA': 'S',    # Serine
    'TCC': 'S',    # Serine
    'TCG': 'S',    # Serine
    'TCT': 'S',    # Serine
    'TTC': 'F',    # Phenylalanine
    'TTT': 'F',    # Phenylalanine
    'TTA': 'L',    # Leucine
    'TTG': 'L',    # Leucine
    'TAC': 'Y',    # Tyrosine
    'TAT': 'Y',    # Tyrosine
    'TAA': 'X',    # Stop
    'TAG': 'X',    # Stop
    'TGC': 'C',    # Cysteine
    'TGT': 'C',    # Cysteine
    'TGA': 'X',    # Stop
    'TGG': 'W',    # Tryptophan
    'CTA': 'L',    # Leucine
    'CTC': 'L',    # Leucine
    'CTG': 'L',    # Leucine
    'CTT': 'L',    # Leucine
    'CCA': 'P',    # Proline
    'CCC': 'P',    # Proline
    'CCG': 'P',    # Proline
    'CCT': 'P',    # Proline
    'CAC': 'H',    # Histidine
    'CAT': 'H',    # Histidine
    'CAA': 'Q',    # Glutamine
    'CAG': 'Q',    # Glutamine
    'CGA': 'R',    # Arginine
    'CGC': 'R',    # Arginine
    'CGG': 'R',    # Arginine
    'CGT': 'R',    # Arginine
    'ATA': 'I',    # Isoleucine
    'ATC': 'I',    # Isoleucine
    'ATT': 'I',    # Isoleucine
    'ATG': 'M',    # Methionine (start)
    'ACA': 'T',    # Threonine
    'ACC': 'T',    # Threonine
    'ACG': 'T',    # Threonine
    'ACT': 'T',    # Threonine
    'AAC': 'N',    # Asparagine
    'AAT': 'N',    # Asparagine
    'AAA': 'K',    # Lysine
    'AAG': 'K',    # Lysine
    'AGC': 'S',    # Serine
    'AGT': 'S',    # Serine
    'AGA': 'R',    # Arginine
    'AGG': 'R',    # Arginine
    'GTA': 'V',    # Valine
    'GTC': 'V',    # Valine
    'GTG': 'V',    # Valine
    'GTT': 'V',    # Valine
    'GCA': 'A',    # Alanine
    'GCC': 'A',    # Alanine
    'GCG': 'A',    # Alanine
    'GCT': 'A',    # Alanine
    'GAC': 'D',    # Aspartic Acid
    'GAT': 'D',    # Aspartic Acid
    'GAA': 'E',    # Glutamic Acid
    'GAG': 'E',    # Glutamic Acid
    'GGA': 'G',    # Glycine
    'GGC': 'G',    # Glycine
    'GGG': 'G',    # Glycine
    'GGT': 'G'     # Glycine
}

AMINO_ACID_TO_CODON = {
    'M': {'ATG': 35}, 
    'I': {'ATA': 32, 'ATC': 33, 'ATT': 34}, 
    'L': {'TTA': 6, 'TTG': 7, 'CTA': 16, 'CTC': 17, 'CTG': 18, 'CTT': 19}, 
    'V': {'GTA': 48, 'GTC': 49, 'GTG': 50, 'GTT': 51}, 
    'F': {'TTC': 4, 'TTT': 5}, 
    'C': {'TGC': 12, 'TGT': 13}, 
    'A': {'GCA': 52, 'GCC': 53, 'GCG': 54, 'GCT': 55}, 
    'G': {'GGA': 60, 'GGC': 61, 'GGG': 62, 'GGT': 63},
    'P': {'CCA': 20, 'CCC': 21, 'CCG': 22, 'CCT': 23}, 
    'T': {'ACA': 36, 'ACC': 37, 'ACG': 38, 'ACT': 39}, 
    'S': {'TCA': 0, 'TCC': 1, 'TCG': 2, 'TCT': 3, 'AGC': 44, 'AGT': 45}, 
    'Y': {'TAC': 8, 'TAT': 9}, 
    'W': {'TGG': 15}, 
    'Q': {'CAA': 26, 'CAG': 27}, 
    'N': {'AAC': 40, 'AAT': 41}, 
    'H': {'CAC': 24, 'CAT': 25}, 
    'E': {'GAA': 58, 'GAG': 59}, 
    'D': {'GAC': 56, 'GAT': 57}, 
    'K': {'AAA': 42, 'AAG': 43}, 
    'R': {'CGA': 28, 'CGC': 29, 'CGG': 30, 'CGT': 31, 'AGA': 46, 'AGG': 47}, 
    'X': {'TAA': 10, 'TAG': 11, 'TGA': 14}, 
}

# this is quicker than creating empty globals, interesting
def generate_empty_kmers(temp, bases, k, prefix):
    """
    Generates all possible kmers of length k and stores them in temp dictionary
    eg: k = 2, bases = ['A', 'T', 'G', 'C'], prefix = ""
        temp = {'AA': 0, 'AT': 0, 'AG': 0, 'AC': 0, 'TA': 0, 'TT': 0, 'TG': 0, 'TC': 0,
                'GA': 0, 'GT': 0, 'GG': 0, 'GC': 0, 'CA': 0, 'CT': 0, 'CG': 0, 'CC': 0}
    :param temp:
    :param bases:
    :param k:
    :param prefix:
    :return:
    """
    if k == 0:
        temp[prefix] = 0
        return
    for base in bases:
        generate_empty_kmers(temp, bases, k - 1, prefix + base)


def get_k_mer_counts(seq, k, overlap, normalize=False):
    kmer_counts = OrderedDict()
    generate_empty_kmers(kmer_counts, ['A', 'T', 'G', 'C'], k, "")
    assert len(kmer_counts) == 4 ** k
    step = 1 if overlap else k
    kmers = [str(seq[i:i + k]) for i in range(0, len(seq), step)]
    for kmer in kmers:
        if len(kmer) < k:
            break
        try:
            kmer_counts[kmer] = kmer_counts[kmer] + 1
        except KeyError:
            pass
    if normalize:
        total = sum(kmer_counts.values())
        if total != 0:
            for key, value in kmer_counts.items():
                kmer_counts[key] = value / total

    return kmer_counts

def nucleobase_percent(seq, tx_size):
    bases = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    if tx_size == 0:
        return bases
    for char in seq:
        if char == 'N':
            continue
        bases[char] += 1
    for key, value in bases.items():
        bases[key] = value / tx_size
    return bases

def wobble_percent(seq):
    bases = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
    total = 0
    for i in range(2, len(seq), 3):
        bases[seq[i]] += 1
        total += 1
    for key, value in bases.items():
        bases[key] = value / total
    return bases



dicodons_to_include = ['AGGCGA', 'AGGCGG', 'ATACGA', 'ATACGG', 'CGAATA', 
                       'CGACCG', 'CGACGA', 'CGACGG', 'CGACTG', 'CGAGCG', 
                       'CTCATA', 'CTCCCG', 'CTGATA', 'CTGCCG', 'CTGCGA', 
                       'CTGCTG', 'CTTCTG', 'GTACCG', 'GTACGA', 'GTGCGA']
def dicodon_counts(seq):
    dicodon_counts = get_k_mer_counts(seq, 6, False, False)
    filtered = {dicodon: dicodon_counts[dicodon] for dicodon in dicodons_to_include}
    return filtered

def kozak_seq(utr5_seq, cds_seq):
    kozak_positions = [-3, -2, -1, 4, 5]
    kozaks = {f'kozak{kozak_position}_{base}': 0 for kozak_position in kozak_positions for base in ['A', 'T', 'G', 'C']}

    for pos in kozak_positions:
        for base in ['A', 'T', 'G', 'C']:
            if pos < 0:
                if len(utr5_seq) >= abs(pos):
                    kozaks[f'kozak{pos}_{base}'] = 1 if utr5_seq[pos] == base else 0
            else:
                kozaks[f'kozak{pos}_{base}'] = 1 if cds_seq[pos-1] == base else 0

    return kozaks


def get_amino_acid_counts(seq, normalize):
    amino_acid_counts = OrderedDict()
    for amino_acid in AMINO_ACID_TO_CODON.keys():
        amino_acid_counts[amino_acid] = 0
    codon_counts = get_k_mer_counts(seq, 3, False, False)
    for codon, count in codon_counts.items():
        amino_acid = CODON_TO_AMINO_ACID[codon]
        amino_acid_counts[amino_acid] += count
    if normalize:
        total = sum(amino_acid_counts.values())
        if total != 0:
            for key, value in amino_acid_counts.items():
                amino_acid_counts[key] = value / total

    return amino_acid_counts


# L - lengths of utr5, cds, utr3 and total sequence
# LL - lengths of utr5, cds, utr3 and total sequence in log scale
# C - codon counts of entire coding sequence
# CF - codon frequencies of entire coding sequence
# P - percentage of each nucleotide in the entire sequence
# P5 - percentage of each nucleotide in the utr5
# PC - percentage of each nucleotide in the cds
# P3 - percentage of each nucleotide in the utr3
# K - nucleotide at the Kozak -3, -2, -1, +4, +5 positions
# 'k'mer5 - 'k'mer counts of utr5, max k is 9, include 'no' for no overlap (6merno5), ex: 6mer5 = 6mer counts of utr5
# 'k'merC - 'k'mer counts of cds, max k is 9, include 'no' for no overlap, ex: 6merC = 6mer counts of cds
# 'k'mer3 - 'k'mer counts of utr3, max k is 9, include 'no' for no overlap, ex: 6mer3 = 6mer counts of utr3
# include freq in kmer to get frequencies instead of counts, ex: 6mer_freq5 = 6mer frequencies of utr5
# WP - percent each nucletide is in wobble position of cds
# DC - dicodon counts of cds
# Biochem - Biochemical features
# Struct - Structural features
def feature_list_from_seq(features_to_extract, seq, utr5_size, cds_size, utr3_size, tx_size):
    temp = []
    if 'L' in features_to_extract:
        temp.append(utr5_size)
        temp.append(cds_size)
        temp.append(utr3_size)
        temp.append(tx_size)

    if 'LL' in features_to_extract:
        temp.append(np.log10(utr5_size) if utr5_size != 0 else 0)
        temp.append(np.log10(cds_size) if cds_size != 0 else 0)
        temp.append(np.log10(utr3_size) if utr3_size != 0 else 0)
        temp.append(np.log10(tx_size) if tx_size != 0 else 0)

    if 'P' in features_to_extract:
        temp = temp + list(nucleobase_percent(seq, tx_size).values())

    utr5_seq = seq[0:utr5_size]
    cds_seq = seq[utr5_size:cds_size+utr5_size]
    utr3_seq = seq[cds_size+utr5_size:tx_size]

    assert len(utr5_seq) == utr5_size
    assert len(cds_seq) == cds_size
    assert len(utr3_seq) == utr3_size
    # print(utr5_seq)
    # print(cds_seq)
    # assert False

    if 'P5' in features_to_extract:
        temp = temp + list(nucleobase_percent(utr5_seq, utr5_size).values())

    if 'PC' in features_to_extract:
        temp = temp + list(nucleobase_percent(cds_seq, cds_size).values())

    if 'P3' in features_to_extract:
        temp = temp + list(nucleobase_percent(utr3_seq, utr3_size).values())

    if 'WP' in features_to_extract:
        temp = temp + list(wobble_percent(cds_seq).values())

    if 'C' in features_to_extract:
        temp = temp + list(get_k_mer_counts(cds_seq, 3, False, False).values())

    if 'CF' in features_to_extract:
        # do include stop codon, diff codons
        temp = temp + list(get_k_mer_counts(cds_seq, 3, False, True).values())

    if 'DC' in features_to_extract:
        temp = temp + list(dicodon_counts(cds_seq).values())

    if 'K' in features_to_extract:
        temp = temp + list(kozak_seq(utr5_seq, cds_seq).values())

    if 'AA' in features_to_extract:
        temp = temp + list(get_amino_acid_counts(cds_seq, False).values())

    if 'AAF' in features_to_extract:
        # don't include stop codon, just a proxy for cds_length
        temp = temp + list(get_amino_acid_counts(cds_seq[0:-3], True).values())
    
    kmers = [i for i in features_to_extract if 'mer' in i]
    for kmer in kmers:
        k = int(kmer[0])
        seq = ""
        if '5' == kmer[-1]:
            seq = utr5_seq
        elif 'C' == kmer[-1]:
            seq = cds_seq
        elif '3' == kmer[-1]:
            seq = utr3_seq
        overlap = not 'no' in kmer
        normalize = 'freq' in kmer
        kmer_counts = get_k_mer_counts(seq, k, overlap, normalize)
        temp = temp + list(kmer_counts.values())
   
    return temp