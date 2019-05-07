import os
import ipdb
import tqdm

if __name__ == '__main__':
    dirnames = os.listdir('/home/litchi/zhuque/lipread_mp4')
    for idx, dirname in tqdm.tqdm(enumerate(dirnames)):
        if idx > 49 and idx < 100:
            #ipdb.set_trace()
            cmd = 'cp -r /home/litchi/zhuque/lipread_mp4/'+dirname+' /home/litchi/zhuque/expdata'
            os.system(cmd)
