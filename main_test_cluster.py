#short = ["adult","alarm", "asia", "child", "grid", "gridr", "insurance", "ring"]
#mid = ["news", "mnist12", "census", "credit"]
#long = ["covtype", "intrusion",  "mnist28"]

PATH = '/home/bigdyl/jindunh/CTGAN_local/result/'
import argparse

parser = argparse.ArgumentParser("ctgan with odes")
parser.add_argument('--data', type=str, default='census')
parser.add_argument('--method', type=int, default = 1)
parser.add_argument('--data_load', type=str, default=PATH + 'G_adult_best_model_Oct03_19-47-10.pth')
args = parser.parse_args()

if args.method == 1:
    from synthesizers._1_ctgan_original_cluster import CTGANSynthesizer as CT # CTGAN
elif args.method == 2:
        from synthesizers.real import CTGANSynthesizer as CT # ODE-CTGAN
elif args.method == 3:
    from synthesizers.real_cluster import CTGANSynthesizer as CT 
elif args.method == 4:
    from synthesizers.real_ab import CTGANSynthesizer as CT
elif args.method == 5:
    from synthesizers.real_ab2 import CTGANSynthesizer as CT
else:
    from synthesizers.real_ab3 import CTGANSynthesizer as CT

syn = CT()
syn.model_load(PATH+args.data_load,args.data)
table = syn.model_test(10, args.data)
group = table['silhouette'].groupby(table.index)
mean = group.mean()
print(args.data)
print(mean)
