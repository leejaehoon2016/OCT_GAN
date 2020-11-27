import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3" 

from synthesizers.adjoint_C import CTGANSynthesizer as OCTGAN
from synthesizers._1_ctgan_original import CTGANSynthesizer as TGAN
from synthesizers.real_ab3 import CTGANSynthesizer as CTGAN_AB3

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
time = time.strftime("%b%d_%H-%M-%S", time.localtime(time.time()))

synthesizer_oct = OCTGAN()
synthesizer_ct = TGAN()
synthesizer_oct3 = CTGAN_AB3()

synthesizer_oct.model_load("/home/jayoung/test/result/G_adult_best_model_Oct03_19-47-10.pth","adult")
synthesizer_ct.model_load("/home/jayoung/test/result/G_adult_best_model_Oct14_02-17-46.pth","adult")
synthesizer_oct3.model_load("/home/jayoung/test/result/G_adult_best_model_Oct13_09-11-55.pth","adult")

feature_names = ["age", "workclass", "fnlwgt", "education", "education-num", "martial-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"] # adult

num_alpha = 101
for i in range(1, 100):
    
    table_oct, jy, syn_oct = synthesizer_oct.model_test(1, "adult", num_alpha=num_alpha) 
    table_ct, syn_ct = synthesizer_ct.model_test(1, "adult", jy)
    table_oct3, syn_oct3 = synthesizer_oct3.model_test(1, "adult", jy)
    
    syn_oct_pd = pd.DataFrame(syn_oct)
    syn_ct_pd = pd.DataFrame(syn_ct)
    syn_oct3_pd = pd.DataFrame(syn_oct3)
    # import pdb;pdb.set_trace()
    os.mkdir("./plots/{}".format(i))

    for col in syn_oct_pd.columns: 
        fig, axes = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(6.5, 3.5))

        # TGAN
        c = axes[0].pcolor(pd.DataFrame(syn_ct_pd.iloc[:, col]).transpose())
        axes[0].set_title('TGAN', fontsize=14)
        axes[0].set_xticks([0, num_alpha], minor=False)
        axes[0].set_yticks([], minor=False)
        axes[0].set_xticklabels(["z1", "z2"], size=14)

        # OCT-GAN ablation3 (w/o ode in G)
        axes[1].pcolor(pd.DataFrame(syn_oct3_pd.iloc[:, col]).transpose())
        axes[1].set_title('OCT-GAN(only_D)', fontsize=14)
        axes[1].set_xticks([0, num_alpha], minor=False)
        axes[1].set_yticks([], minor=False)
        axes[1].set_xticklabels(["z1", "z2"], size=14)
        
        # OCT-GAN
        axes[2].pcolor(pd.DataFrame(syn_oct_pd.iloc[:, col]).transpose())
        axes[2].set_title('OCT-GAN', fontsize=14)
        axes[2].set_xticks([0, num_alpha], minor=False)
        axes[2].set_yticks([], minor=False)
        axes[2].set_xticklabels(["z1", "z2"], size=14)

        fig.tight_layout()
        fig.subplots_adjust(top=0.75)
        fig.suptitle(feature_names[col], fontsize=20)

        cb = fig.colorbar(c, ax=axes.ravel().tolist(), shrink=1, aspect=6)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(14)

        plt.savefig("./plots/{}/plot_{}.png".format(i, col), dpi=300)
