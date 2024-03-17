from Diffusion.gaussian_diffusion import GaussianDiffusion1D
from Diffusion.unet_1d import Unet1D
from Diffusion.modules_1D import Dataset1D
from Diffusion.diffusion_trainer import Trainer1D

from Protein.protein_utrs import Protein_UTR


protein_seq = "CPEEWEWNRSVMSVHNLCWQQAVDLGLWWILVPMIGGMIYMRQPLHRWLASSFKVFAIYVSIGGQVKRWPVVRFYSMEVWDYLWGYNYYELCIVKCGNYEEKLNIYTDMNRANWPLQFKSWKGGFKGSQYKHAKGTQLRGVSWSRRDTGFCDTMRMRLDWKISWTKHAMIQQRRLFQCSVKFKCFAIGGKEKWWCPMGGKHRGEPLPPKNYCPMVEHYIWFWYFGLFVKRRQDNTRLQKLICLILDNFPCIDNNYDTCYTIEMPDLLCATEQNQCRDMDCYKHPREACIECEGCEPDTWGVSDNTNNKFGICFHRTPQKGLQSTEEIRGDPRGLYKTRGGLMDGWYVNAYFHFTQFHFYDWLEKCCMGIFQEYCMVHEYHANVIIGKVYRQQMCPGYYWKTAMPKFWWHIFNLPSKEITQFIKEVNQYLESQSDTKIKCEAKKGTRRLSFLNCVLLELYCDRDIQMECQRWVRKPWHNQHFSNLRFAGTYSWDQQLRYNTATAAVIKNTASVFTEWCRDLSKTPAMGRFATEAKAGNFKAWKMAHCKRVAPLKKMCQFEFQDVSNWAEFVRDWEFSHREWRAEFVNDLIPDINKLPQSSNTHISNKCYDQNQWTIMIEHAQPMDYMHTGQIKKVMSVGHGMYYPHCISQITWINSFIDTANTKDDHMPSQQRVPSTTSNEHKRYVAMFFSVVYGNTKFNWGNPGHHKPHAPLHTALQNFNTFFFAYTVPGRMHYWWHHVHYLWLPDFWCLCSMKDWCHHSQSKRYGVPLSQYEVDGCQDVWRMQKNMDTQFVLNWLDSGRAQGSACTEINPCPKVKMNSPCQNFHSRMWFRMRKPHLGVEFLIPNDGAKNFFLVDFCIFMMGCCMSRNVKPVMGTPCPHMYLSNHQTVQLIMDQNRFQERAIWYANDRQIDWLHNAVETTAYTYTTWRHEGHLDVLRADVVMWHFSWDVFYYCVQWFQIMNWFHDNGNVHLVSWYLSNAAYKEYSFFVTMQMKAPVQSIS"
protein_obj = Protein_UTR(protein_seq, 497, 83)

diffusion = GaussianDiffusion1D(
    model = Unet1D(
            dim = 32,
            dim_mults = (1, 2, 5),
            channels = 1,
        ),
    seq_length = len(protein_seq)*3+497+83,
    timesteps = 1000,
    objective = 'pred_v'
)

training_seq = protein_obj.generate_n_sequences(10000).half()
print(training_seq)
print(protein_obj.multi_check_sequence(training_seq))
dataset = Dataset1D(training_seq) 

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 1e-3,
    train_num_steps = 600000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every=1000,
    results_folder="utr_diffusion",
    save_training_data=True,
    protein = protein_obj
)

if __name__ == '__main__':
    trainer.train()