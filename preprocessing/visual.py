
import random
from itertools import count
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets
from ipywidgets import interact

# %matplotlib qt
matplotlib.use('TkAgg')
plt.ion()
class loss_precision_recall():
    def __init__(self, epochs, labels, loss_rng):
        self.epochs = epochs
        self.cur_trial = 0
        self.labels = labels
        self.num_labels = len(labels)
        self.loss_rng = loss_rng
        self.fig = plt.figure(figsize=(12, 15))
        self.data = {
            'losses': [],  # trials x epochs (per trial)
            'label_pr': [] # trials x labels x 2 (p & r) x epochs
        }
        
    def report(self, loss, lables_pr):
        """
        labels_pr shape = labels x [p, r]
        """
        self.data['losses'][self.cur_trial - 1].append(loss)
        for label in range(self.num_labels):
            for pr in range(2):
                self.data['label_pr'][self.cur_trial - 1][label][pr].append(lables_pr[pr][label])
                
        plt.clf()
        self.plot()
        
    def new_trial(self):
        self.data['losses'].append( [ ] )
        self.data['label_pr'].append([ [ [] , [] ] for i in range(self.num_labels) ])
        self.cur_trial += 1
        
    def start(self):
        pass
        # plt.figure(figsize=(12, 15))
        
    def plot(self):
        # plt.clf()
        fig = self.fig
        gs = fig.add_gridspec(7, 2, width_ratios=[0.8, 1], left=0.05, right=0.95, height_ratios=[2]*7, hspace=2)

        ax1 = fig.add_subplot(gs[:, 0])
        ax1.set_xlabel('Epochs', labelpad=5)
        ax1.set_ylabel('Loss', labelpad=5)
        ax1.set_xlim(0, self.epochs)
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.set_ylim(0, self.loss_rng)

        for trial in range(self.cur_trial):
            color = "blue"
            if trial == self.cur_trial - 1:
                color = "red"
            y = self.data['losses'][trial]
            x = list(range(len(y)))
            ax1.plot(x, y, color=color)

        # Right plots (7 stacked on the right side)
        for label in range(7):
            ax2 = fig.add_subplot(gs[label, 1])
            ax2.set_xlabel(f'Epochs\n{self.labels[label]}', labelpad=0)
            ax2.set_ylabel('Precision\nrecall', labelpad=0)
            ax2.set_xlim(0, self.epochs)
            ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax2.set_ylim(0, 1.0)
            
            p_color = "green"
            r_color = "blue"
            for trial in range(self.cur_trial):
                if trial == self.cur_trial - 1: 
                    p_color = "yellow"
                    r_color = "purple"
                precision_y = self.data['label_pr'][trial][label][0]
                recall_y = self.data['label_pr'][trial][label][1]
                x = list(range(len(recall_y)))
                ax2.plot(x, precision_y, color=p_color)
                ax2.plot(x, recall_y, color=r_color)
                
        # plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)

def plot_cross_section(*args):
    colors =['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    def plotter(i):
        plt.figure(figsize=(15, 5))
        alpha1 = 0.3
        
        plt.subplot(131)
        for i, arg in enumerate(args):
            # Slice at x-coordinate
            plt.imshow(arg[i, :, :], cmap=cmaps[i], alpha=alpha1)
            plt.title(f'Slice at x={i}')

        # Slice at y-coordinate
        plt.subplot(132)
        for i, arg in enumerate(args):
            plt.imshow(arg[:, i, :], cmap=cmaps[i], alpha=alpha1)
            plt.title(f'Slice at y={i}')

        # Slice at z-coordinate
        plt.subplot(133)
        for i, arg in enumerate(args):
            plt.imshow(arg[:, :, i], cmap=cmaps[i], alpha=alpha1)
            plt.title(f'Slice at z={i}')

        plt.show()

    # Interactive Slider for scrolling through slices
    interact(plotter, i=(0, args[0].shape[0] - 1))