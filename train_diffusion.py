from Diffusion.gaussian_diffusion import GaussianDiffusion1D
from Diffusion.unet_1d import Unet1D
from Diffusion.modules_1D import Dataset1D
from Diffusion.diffusion_trainer import Trainer1D

from Protein.protein import Protein


protein_seq = "CPEEWEWNRSVMSVHNLCWQQAVDLGLWWILVPMIGGMIYMRQPLHRWLASSFKVFAIYVSIGGQVKRWPVVRFYSMEVWDYLWGYNYYELCIVKCGNYEEKLNIYTDMNRANWPLQFKSWKGGFKGSQYKHAKGTQLRGVSWSRRDTGFCDTMRMRLDWKISWTKHAMIQQRRLFQCSVKFKCFAIGGKEKWWCPMGGKHRGEPLPPKNYCPMVEHYIWFWYFGLFVKRRQDNTRLQKLICLILDNFPCIDNNYDTCYTIEMPDLLCATEQNQCRDMDCYKHPREACIECEGCEPDTWGVSDNTNNKFGICFHRTPQKGLQSTEEIRGDPRGLYKTRGGLMDGWYVNAYFHFTQFHFYDWLEKCCMGIFQEYCMVHEYHANVIIGKVYRQQMCPGYYWKTAMPKFWWHIFNLPSKEITQFIKEVNQYLESQSDTKIKCEAKKGTRRLSFLNCVLLELYCDRDIQMECQRWVRKPWHNQHFSNLRFAGTYSWDQQLRYNTATAAVIKNTASVFTEWCRDLSKTPAMGRFATEAKAGNFKAWKMAHCKRVAPLKKMCQFEFQDVSNWAEFVRDWEFSHREWRAEFVNDLIPDINKLPQSSNTHISNKCYDQNQWTIMIEHAQPMDYMHTGQIKKVMSVGHGMYYPHCISQITWINSFIDTANTKDDHMPSQQRVPSTTSNEHKRYVAMFFSVVYGNTKFNWGNPGHHKPHAPLHTALQNFNTFFFAYTVPGRMHYWWHHVHYLWLPDFWCLCSMKDWCHHSQSKRYGVPLSQYEVDGCQDVWRMQKNMDTQFVLNWLDSGRAQGSACTEINPCPKVKMNSPCQNFHSRMWFRMRKPHLGVEFLIPNDGAKNFFLVDFCIFMMGCCMSRNVKPVMGTPCPHMYLSNHQTVQLIMDQNRFQERAIWYANDRQIDWLHNAVETTAYTYTTWRHEGHLDVLRADVVMWHFSWDVFYYCVQWFQIMNWFHDNGNVHLVSWYLSNAAYKEYSFFVTMQMKAPVQSIS"
protein_obj = Protein(protein_seq)

diffusion = GaussianDiffusion1D(
    model = Unet1D(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = 3,
        ),
    seq_length = len(protein_seq),
    timesteps = 1000,
    sampling_timesteps = 500,
    objective = 'pred_v'
)

training_seq = protein_obj.generate_n_sequences(10000).half()
dataset = Dataset1D(training_seq) 

trainer = Trainer1D(
    diffusion,
    dataset = dataset,
    train_batch_size = 32,
    train_lr = 3e-2,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    save_and_sample_every=1000,
    results_folder="ddim_diffusion",
    save_training_data=True
)

if __name__ == '__main__':
    trainer.train()