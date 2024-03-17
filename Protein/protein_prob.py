from Bio.Data import CodonTable
import random
import numpy as np
import torch as t

from LGBM.lgbm import LGBM_TE_model

class Protein_Prob:
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

    def __construct_color_mapping(self, A=0, U=1, C=2, G=3):
        self.base_mapping = {
            'A': A,
            'U': U,
            'C': C,
            'G': G,
             A: 'A',
             U: 'T',
             C: 'C',
             G: 'G',
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

    def __onehot_encode(self, base):
        encoding = [0,0,0,0]
        encoding[self.base_mapping[base]] = 1
        return encoding

    def __decode_onehot(self, onehot):
        seq = onehot.transpose()
        index_sequence = np.argmax(seq, axis=1)
        dna_seq = np.array([self.base_mapping[idx] for idx in index_sequence])
        return ''.join(dna_seq)

    def __generate_random_sequence(self):
        seq = []
        for amino_acid in self.sequence:
            try:
                codon = random.choice(self.__inverse_table[amino_acid])
                for base in codon:
                    seq.append(self.__onehot_encode(base))
            except KeyError:
                print("Amino Acid not in codon table")
                return None
        return np.array(seq, dtype=np.double).transpose()
    
    def generate_n_sequences(self, n):
        seqs = []
        for _ in range(n):
            seqs.append(self.__generate_random_sequence())
        return t.from_numpy(np.array(seqs))
    
    def sequence_generator(self):
        yield t.from_numpy(np.array(self.__generate_random_sequence()))

    # Returns the number of bases that are off for a single sequence. Takes in a torch
    def validate_sequence(self, sequence):
        dna_seq = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        mismatch = 0
        for codon, amino_acid in zip(dna_seq, self.sequence):
            try:
                if not (self.codon_table.forward_table[codon] == amino_acid):
                    mismatch+=1
            except KeyError:
                mismatch+=1

        return mismatch

    def __single_maximize_base(self, sequence, base='G'):
        cds = self.__decode_onehot(sequence)
        mismatch = self.validate_sequence(cds)
        return sequence.count(base) if mismatch==0 else -mismatch

    def multi_maximize_base(self, sequences, base = 'G'):
        count = []
        sequences = sequences.cpu().numpy()
        for sequence in sequences:
            count.append(self.__single_maximize_base(sequence, base))
        return t.Tensor(count)

    def __single_check_sequence(self, sequence):
        cds = self.__decode_onehot(sequence)
        return self.validate_sequence(cds)

    def multi_check_sequence(self, sequences):
        count = []
        sequences = sequences.cpu().numpy()
        for sequence in sequences:
            print(sequence)
            count.append(self.__single_check_sequence(sequence))
        return t.Tensor(count)

    def __single_reward_TE_prediction(self, sequence):
        cds = self.__decode_onehot(sequence)
        mismatch = self.validate_sequence(cds)
        return (self.LGBM_model.predict_TE(cds)[0] if mismatch==0 else -mismatch)*self.SCALE_FACTOR

    def multi_reward_TE_prediction(self, sequences):
        count = []
        seqs = sequences.cpu().numpy()
        for sequence in seqs:
            count.append(self.__single_reward_TE_prediction(sequence))
        return t.Tensor(count)