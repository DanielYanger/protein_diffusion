from Bio.Data import CodonTable
import random
import numpy as np
import torch as t

from LGBM.lgbm_replace import LGBM_TE_model

class Protein_UTR:
    def __init__(self, 
                sequence: str, 
                utr3_len,
                utr5_len,
                models_dir: str = '/work/09360/dayang/ls6/protein_diffusion/LGBM/LL_P5_P3_CF_AAF_3mer_freq_5',
                codon_table = CodonTable.standard_dna_table):

        
        self.sequence = sequence
        self.codon_table = codon_table

        self.__construct_color_mapping()
        self.__construct_inverse_table()

        self.utr3_len = utr3_len
        self.utr5_len = utr5_len
        self.coding_sequence = self.__generate_coding_region()
        self.cds_len = len(self.coding_sequence)
        self.total_length = utr3_len+utr5_len+self.cds_len
        

        self.LGBM_model = LGBM_TE_model(models_dir, utr5_len, utr3_len)
        self.SCALE_FACTOR = 50

    def __construct_color_mapping(self, A=1, T=2, C=3, G=4):
        self.base_mapping = {
            'A': A,
            A: 'A',
            'T': T,
            T: 'T',
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
                    try:
                        self.__inverse_table[amino].append(codon)
                    except KeyError:
                        self.__inverse_table[amino] = [codon]
        self.__inverse_table['B'] = self.__inverse_table['N'] + self.__inverse_table['D']
        self.__inverse_table['Z'] = self.__inverse_table['Q'] + self.__inverse_table['E']
        print(self.__inverse_table)

    def __generate_coding_region(self):
        seq = ""
        for amino_acid in self.sequence:
            try:
               seq+=self.__inverse_table[amino_acid][0] 
            except KeyError:
                print("Amino Acid not in codon table")
        return seq

    def __generate_random_sequence(self):
        seq = ""

        for _ in range(self.utr5_len):
            seq+=(random.choice("ACTG"))

        seq+=self.coding_sequence

        for _ in range(self.utr3_len):
            seq+=(random.choice("ACTG"))
        
        return seq
    
    def __encode_random_sequence(self, sequence):
        seq = [[]]
        for base in sequence:
            seq[0].append(self.base_mapping[base])
        
        return np.array(seq, dtype=np.double)/4.0
    
    def generate_n_sequences(self, n):
        seqs = []
        for _ in range(n):
            seqs.append(self.__encode_random_sequence(self.__generate_random_sequence()))
        return t.from_numpy(np.array(seqs))
    
    def sequence_generator(self):
        yield t.from_numpy(np.array(self.__encode_random_sequence(self.__generate_random_sequence())))
    
    def multi_check_sequence(self, seq):
        count = []
        seq = (seq.cpu().numpy()*4).round()
        for sequence in seq:
            sequence = sequence[0][self.utr5_len:self.utr5_len+self.cds_len]
            print(sequence)
            internal_count = 0
            incorrect = 0
            cds = ""
            
            for generated_base, true_base, i in zip(sequence, self.coding_sequence, range(len(sequence))):
                try:
                    if self.base_mapping[generated_base]!=true_base:
                        incorrect+=1
                except:
                    incorrect+=1
                    
            count.append(incorrect)
        
        return t.Tensor(count)

    def reward_TE_prediction(self, sequence):
        count = []
        # seq = np.clip(sequence.cpu().numpy()*4, 1, 4).round()
        seq = (sequence.cpu().numpy()*4).round()
        for sequence in seq:
            cds = ''
            for codon in sequence[0]:
                try:
                    cds+=self.base_mapping[codon]
                except KeyError:
                    break
            if len(cds) != self.total_length:
                count.append(-10 * self.SCALE_FACTOR)
                continue

            # cds[self.utr5_len:self.utr5_len+self.cds_len] = self.coding_sequence
            # print('utr3 ',cds[0:self.utr5_len])
            # print('utr5', cds[self.utr5_len+self.cds_len:])
            count.append(self.LGBM_model.predict_TE(cds[0:self.utr5_len]+self.coding_sequence+cds[self.utr5_len+self.cds_len:])[0] * self.SCALE_FACTOR)
        
        return t.Tensor(count)