import os
import ipdb

if __name__ == '__main__':
    dirnames = os.listdir('/home/litchi/zhuque/lipread_mp4')
    for idx, dirname in enumerate(dirnames):
        if idx > 49 and idx < 101:
            #ipdb.set_trace()
            cmd = 'cp -r /home/litchi/zhuque/lipread_mp4/'+dirname+' /home/litchi/zhuque/expdata'
            os.system(cmd)
