from Diffusion.gaussian_diffusion import GaussianDiffusion1D
from Diffusion.unet_1d import Unet1D
from Diffusion.modules_1D import Dataset1D
from Diffusion.diffusion_trainer import Trainer1D

from Protein.protein_utrs import Protein_UTR


protein_seq = "MPEEWEWNRSVMSVHNLCWQQAVDLGLWWILVPMIGGMIYMRQPLHRWLASSFKVFAIYVSIGGQVKRWPVVRFYSMEVWDYLWGYNYYELCIVKCGNYEEKLNIYTDMNRANWPLQFKSWKGGFKGSQYKHAKGTQLRGVSWSRRDTGFCDTMRMRLDWKISWTKHAMIQQRRLFQCSVKFKCFAIGGKEKWWCPMGGKHRGEPLPPKNYCPMVEHYIWFWYFGLFVKRRQDNTRLQKLICLILDNFPCIDNNYDTCYTIEMPDLLCATEQNQCRDMDCYKHPREACIECEGCEPDTWGVSDNTNNKFGICFHRTPQKGLQSTEEIRGDPRGLYKTRGGLMDGWYVNAYFHFTQFHFYDWLEKCCMGIFQEYCMVHEYHANVIIGKVYRQQMCPGYYWKTAMPKFWWHIFNLPSKEITQFIKEVNQYLESQSDTKIKCEAKKGTRRLSFLNCVLLELYCDRDIQMECQRWVRKPWHNQHFSNLRFAGTYSWDQQLRYNTATAAVIKNTASVFTEWCRDLSKTPAMGRFATEAKAGNFKAWKMAHCKRVAPLKKMCQFEFQDVSNWAEFVRDWEFSHREWRAEFVNDLIPDINKLPQSSNTHISNKCYDQNQWTIMIEHAQPMDYMHTGQIKKVMSVGHGMYYPHCISQITWINSFIDTANTKDDHMPSQQRVPSTTSNEHKRYVAMFFSVVYGNTKFNWGNPGHHKPHAPLHTALQNFNTFFFAYTVPGRMHYWWHHVHYLWLPDFWCLCSMKDWCHHSQSKRYGVPLSQYEVDGCQDVWRMQKNMDTQFVLNWLDSGRAQGSACTEINPCPKVKMNSPCQNFHSRMWFRMRKPHLGVEFLIPNDGAKNFFLVDFCIFMMGCCMSRNVKPVMGTPCPHMYLSNHQTVQLIMDQNRFQERAIWYANDRQIDWLHNAVETTAYTYTTWRHEGHLDVLRADVVMWHFSWDVFYYCVQWFQIMNWFHDNGNVHLVSWYLSNAAYKEYSFFVTMQMKAPVQSIS"
protein_obj = Protein_UTR(protein_seq, 497, 83)
print("1, 2, 4, ..., 1, 4, 2")
multpliers = (4, 4, 8)
print(multpliers)


diffusion = GaussianDiffusion1D(
    model = Unet1D(
            dim = 64,
            dim_mults = multpliers,
            channels = 1,
        ),
    seq_length = len(protein_seq)*3+497+83, #3580
    timesteps = 1000,
    objective = 'pred_v'
)

training_seq = protein_obj.generate_n_sequences(10000).half()
dataset = Dataset1D(training_seq) 

lr = 3e-4
print(f"lr : {lr}")


trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = lr,
    train_num_steps = 1000000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every=1000,
    results_folder="utr_diffusion_ddim_sample",
    save_training_data=True,
    protein = protein_obj
)

if __name__ == '__main__':
    trainer.train()