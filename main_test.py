from synthesizers.real import CTGANSynthesizer as CT
# from synthesizers.real_cluster import CTGANSynthesizer as CT
# from synthesizers.real_ab import CTGANSynthesizer as CT
# from synthesizers.real_ab2 import CTGANSynthesizer as CT
# from synthesizers.real_ab3 import CTGANSynthesizer as CT

short = ["adult", "alarm", "asia", "child", "grid", "gridr", "insurance", "ring"]
mid = ["news", "mnist12", "census", "credit"]
long = ["covtype", "intrusion", "mnist28"]

MODEL_PATH = "/home/bigdyl/jindunh/CTGAN_local/result/G_intrusion_L_test_best_model_Oct16_09-58-59.pth"
syn = CT()
syn.model_load(MODEL_PATH, "intrusion")

for i in range(10):
    print(i)
    table = syn.model_test(10, "intrusion") 
    temp = table['macro_f1'].groupby(table['iters'])
    print(temp.mean().mean())
