import os

import torch
from Protein.protein_prob import Protein_Prob
from Protein.protein import Protein

def count_files_in_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)
    # Count the number of files
    return len(files)

protein_seq = "CPEEWEWNRSVMSVHNLCWQQAVDLGLWWILVPMIGGMIYMRQPLHRWLASSFKVFAIYVSIGGQVKRWPVVRFYSMEVWDYLWGYNYYELCIVKCGNYEEKLNIYTDMNRANWPLQFKSWKGGFKGSQYKHAKGTQLRGVSWSRRDTGFCDTMRMRLDWKISWTKHAMIQQRRLFQCSVKFKCFAIGGKEKWWCPMGGKHRGEPLPPKNYCPMVEHYIWFWYFGLFVKRRQDNTRLQKLICLILDNFPCIDNNYDTCYTIEMPDLLCATEQNQCRDMDCYKHPREACIECEGCEPDTWGVSDNTNNKFGICFHRTPQKGLQSTEEIRGDPRGLYKTRGGLMDGWYVNAYFHFTQFHFYDWLEKCCMGIFQEYCMVHEYHANVIIGKVYRQQMCPGYYWKTAMPKFWWHIFNLPSKEITQFIKEVNQYLESQSDTKIKCEAKKGTRRLSFLNCVLLELYCDRDIQMECQRWVRKPWHNQHFSNLRFAGTYSWDQQLRYNTATAAVIKNTASVFTEWCRDLSKTPAMGRFATEAKAGNFKAWKMAHCKRVAPLKKMCQFEFQDVSNWAEFVRDWEFSHREWRAEFVNDLIPDINKLPQSSNTHISNKCYDQNQWTIMIEHAQPMDYMHTGQIKKVMSVGHGMYYPHCISQITWINSFIDTANTKDDHMPSQQRVPSTTSNEHKRYVAMFFSVVYGNTKFNWGNPGHHKPHAPLHTALQNFNTFFFAYTVPGRMHYWWHHVHYLWLPDFWCLCSMKDWCHHSQSKRYGVPLSQYEVDGCQDVWRMQKNMDTQFVLNWLDSGRAQGSACTEINPCPKVKMNSPCQNFHSRMWFRMRKPHLGVEFLIPNDGAKNFFLVDFCIFMMGCCMSRNVKPVMGTPCPHMYLSNHQTVQLIMDQNRFQERAIWYANDRQIDWLHNAVETTAYTYTTWRHEGHLDVLRADVVMWHFSWDVFYYCVQWFQIMNWFHDNGNVHLVSWYLSNAAYKEYSFFVTMQMKAPVQSIS"
protein_obj = Protein_Prob(protein_seq)
# protein_obj2 = Protein(protein_seq)
filepath = "../prob_protein_diffusion/checkpoints/samples/"

if __name__ == '__main__':
    error_plot = []
    individual_error = []

    for i in range(1, count_files_in_directory(filepath)+1):
        tensor = torch.load(f"{filepath}/sample-{i}.pt", map_location=torch.device('cpu'))
        errors = protein_obj.multi_check_sequence(tensor)
        tensor_error = torch.sum(errors).item()
        error_count = torch.sum(errors > 0).item()
        error_plot.append(tensor_error/len(tensor))
        individual_error.append(error_count/len(tensor) * 100.0)
        print(tensor_error/len(tensor))
        print(error_count/len(tensor) * 100.0)
    
    print(error_plot)
    print(individual_error)
