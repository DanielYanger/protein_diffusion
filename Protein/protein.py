from Bio.Data import CodonTable
import random
import numpy as np
import torch as t

from LGBM.lgbm import LGBM_TE_model

class Protein:
    def __init__(self, 
                sequence: str, 
                models_dir: str = '/work/09360/dayang/ls6/protein_diffusion/LGBM/LL_P5_P3_CF_AAF_3mer_freq_5',
                codon_table = CodonTable.standard_dna_table):

        
        self.sequence = sequence
        self.codon_table = codon_table
        self.__construct_color_mapping()
        self.__construct_inverse_table()

        self.LGBM_model = LGBM_TE_model(models_dir)
        self.SCALE_FACTOR = 10

    def __construct_color_mapping(self, A=1, U=2, C=3, G=4):
        self.base_mapping = {
            'A': A,
            A: 'A',
            'U': U,
            U: 'T',
            'C': C,
            C: 'C',
            'G': G,
            G: 'G'
        }

    def __construct_inverse_table(self):
        self.__inverse_table = {}
        for base1 in "ATCG":
            for base2 in "ATCG":
                for base3 in "ATCG":
                    codon = base1 + base2 + base3
                    try:
                        amino = self.codon_table.forward_table[codon]
                    except KeyError:
                        continue
                    codon = codon.replace('T', 'U')
                    try:
                        self.__inverse_table[amino].append(codon)
                    except KeyError:
                        self.__inverse_table[amino] = [codon]
        self.__inverse_table['B'] = self.__inverse_table['N'] + self.__inverse_table['D']
        self.__inverse_table['Z'] = self.__inverse_table['Q'] + self.__inverse_table['E']

    def __generate_random_sequence(self):
        seq = []
        for amino_acid in self.sequence:
            try:
                codon = random.choice(self.__inverse_table[amino_acid])
                seq.append([self.base_mapping[codon[0]], self.base_mapping[codon[1]], self.base_mapping[codon[2]]])
            except KeyError:
                print("Amino Acid not in codon table")
        return np.array(seq, dtype=np.double).transpose()/4.0
    
    def generate_n_sequences(self, n):
        seqs = []
        for _ in range(n):
            seqs.append(self.__generate_random_sequence())
        return t.from_numpy(np.array(seqs))
    
    def sequence_generator(self):
        yield t.from_numpy(np.array(self.__generate_random_sequence()))

    def validate_sequence(self, seq):
        seq = (seq.cpu().numpy().transpose()*4.0).round()
        mismatch = 0
        for codon, amino_acid, i in zip(seq, self.sequence, range(len(seq))):
            try:
                codon_str = self.base_mapping[codon[0]]+self.base_mapping[codon[1]]+self.base_mapping[codon[2]]
                if not self.codon_table.forward_table[codon_str] == amino_acid:
                    mismatch+=1
            except KeyError:
                mismatch+=1
        return mismatch
    
    def verify(self, seq):
        count = []
        seq = (seq.cpu().numpy()*4).round()
        for sequence in seq:
            sequence = sequence.transpose()
            internal_count = 0
            incorrect = False
            for codon, amino_acid, i in zip(sequence, self.sequence, range(len(sequence))):
                try:
                    codon_str = self.base_mapping[codon[0]]+self.base_mapping[codon[1]]+self.base_mapping[codon[2]]
                    if not self.codon_table.forward_table[codon_str] == amino_acid:
                        incorrect = True
                        break
                    else:
                        internal_count+=codon_str.count('G')
                except KeyError:
                    incorrect = True
                    break
            count.append(internal_count if not incorrect else 0)
        
        return t.Tensor(count)

    def reward_TE_prediction(self, sequence):
        count = []
        seq = (sequence.cpu().numpy()*4).round()
        for sequence in seq:
            sequence = sequence.transpose()
            mismatch = 0
            valid_seq = True
            cds = ''
            for codon, amino_acid, i in zip(sequence, self.sequence, range(len(sequence))):
                try:
                    codon_str = self.base_mapping[codon[0]]+self.base_mapping[codon[1]]+self.base_mapping[codon[2]]
                    if not self.codon_table.forward_table[codon_str] == amino_acid:
                        mismatch+=1
                        valid_seq = False

                    if valid_seq:
                        cds+=codon_str
                except KeyError:
                    mismatch+=1
                    valid_seq = False
            count.append((self.LGBM_model.predict_TE(cds)[0] if valid_seq else -mismatch)*self.SCALE_FACTOR)
        return t.Tensor(count)