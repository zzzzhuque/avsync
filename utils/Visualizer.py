#coding=utf-8
import visdom
import torch
import numpy as np
import time

class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        # 横坐标的1,2,3...表示画的第几个loss
        #self.index = {}
        #self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.iteritems():
            '''
            param d: dict (name, value) i.e. ('loss', 0.11)
            '''
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, x, y, win):
        x = torch.DoubleTensor([x])
        self.vis.line(X=x,
                      Y=y,
                      win=unicode(win),
                      opts=dict(title=win),
                      update=None if x==0 else 'append'
                      )
        #self.index[name] = x1

    def img(self, name, img_, **kwargs):
        self.vis.images(img_.cpu().numpy(),
                        win=unicode(name),
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info
        ))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
