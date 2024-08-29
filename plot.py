import os
from data.bot.data import START_DAY, END_DAY, ONE_DAY

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib

def plot_score_sum(ax, user, loc, legend=False):

    benign_user = user.loc[(user.used_bot.eq(False))]
    cheater_user = user.loc[(user.used_bot.eq(True))]

    # Size for number of user with a same value
    #handles, labels = ax.scatter.legend_elements(prop="sizes", alpha=1, num = 4)
    #legend2 = ax.legend(handles, labels, loc="lower right", title="Sizes")
    
    x = benign_user.val 
    y = benign_user.dub 
    #size = benign_user.groupby(['dub','val'])['id'].transform('count') + 50
    #scatter = ax.scatter(x, y, s= size, c='b', label='Normal User')
    scatter = ax.scatter(x, y, c='b', label='Normal User')
    if legend is True:
        ax.legend(loc = "upper right")
        
        
    x = cheater_user.val 
    y = cheater_user.dub 

    color = cheater_user.played_match
    #size = cheater_user.groupby(['dub','val'])['id'].transform('count') + 50
    #scatter = ax.scatter(x, y, s= size, c=color, marker='^', cmap="Dark2", label='Cheater')
    scatter = ax.scatter(x, y, c=color, marker='^', cmap="Dark2", label='Cheater')
    # Color class for cheater or user
    if legend is True:
        legend1 = ax.legend(*scatter.legend_elements(prop="colors", num=9),
                        loc=loc, title="Played match", bbox_to_anchor=(0.5,-0.45), ncol=9, fancybox=True, shadow=True)
        ax.add_artist(legend1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, -0.2), fancybox=True, shadow=True)
        
    ax.set_xlabel('Sum of validity')
    ax.set_ylabel('Sum of dubious')
    
    return ax    

def plot_voted_count(ax, user, legend=False):

    benign_user = user.loc[(user.used_bot.eq(False))]
    cheater_user = user.loc[(user.used_bot.eq(True))]
    '''
    bx = benign_user.voted_to_cheat
    by = benign_user.voted_to_untrue
    
    cx = cheater_user.voted_to_cheat
    cy = cheater_user.voted_to_untrue
    
    def heatmap(data, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current Axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if ax is None:
            ax = plt.gca()

        if cbar_kw is None:
            cbar_kw = {}

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]), labels=np.arange(data.shape[0]))

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                rotation_mode="anchor")

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar


    def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                        textcolors=("black", "white"),
                        threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts
    
    cheater_count = cheater_user.value_counts(['voted_to_untrue', 'voted_to_cheat']) 
    cheater_count = cheater_count.to_frame().reset_index()
    
    benign_count = benign_user.value_counts(['voted_to_untrue', 'voted_to_cheat'])
    benign_count = benign_count.to_frame().reset_index()
    
    total_count = pd.concat([cheater_count, benign_count]).groupby(['voted_to_untrue', 'voted_to_cheat']).sum().reset_index()
    total_count = total_count.pivot(index='voted_to_untrue', columns='voted_to_cheat', values='count')
    full_index = total_count.index.union(total_count.columns)   

    cheater_count = cheater_count.pivot(index='voted_to_untrue', columns='voted_to_cheat', values='count')
    benign_count = benign_count.pivot(index='voted_to_untrue', columns='voted_to_cheat', values='count')

    cheater_count = cheater_count.reindex(labels=full_index, axis=0).reindex(labels=full_index, axis=1)
    benign_count = benign_count.reindex(labels=full_index, axis=0).reindex(labels=full_index, axis=1)

    cheater_ratio = (cheater_count / total_count)
    benign_ratio = (benign_count / total_count)
    
    data = benign_ratio.sub(cheater_ratio, fill_value=0)

    data = data.dropna(axis=0, how='all')
    data = data.dropna(axis=1, how='all')
    im, cbar = heatmap(data.fillna(0.), ax=ax, cmap='YlGn', cbarlabel='Ratio of result\n[Num. of Cheater / Num. of Normal user]')
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    return ax
    '''
    x = benign_user.voted_to_cheat 
    y = benign_user.voted_to_untrue 
    color = benign_user.bot_usage_days
    #size = benign_user.groupby(['dub','val'])['id'].transform('count') + 50
    #scatter = ax.scatter(x, y, s= size, c='b', cmap="Spectral", label='Normal User')
    scatter = ax.scatter(x, y, c='b', cmap="Spectral", label='Normal User')
    
    x = cheater_user.voted_to_cheat
    y = cheater_user.voted_to_untrue
    color = cheater_user.played_match
    size = cheater_user.groupby(['dub','val'])['id'].transform('count') + 50
    
    
    scatter = ax.scatter(x, y, s= size, c=color, marker='^', cmap='Dark2', label='Cheater')    
    
    if legend is True:
        legend1 = ax.legend(*scatter.legend_elements(prop="colors", num=9),
                         loc="lower right", title="Played match", bbox_to_anchor=(0.05,-0.5), ncol=5, fancybox=True, shadow=True)
        ax.add_artist(legend1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, -0.18), fancybox=True, shadow=True)
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_xlabel('Ended up with high dubious (<0.0)')
    ax.set_ylabel('Ended up with low validity (<0.5)')
    
    return ax
    #plt.savefig(os.path.join('img', filename), bbox_inches='tight', pad_inches=0)


def figure1(moba_without_liar, moba_with_liar, action_without_liar, action_with_liar):
    fig, axs = plt.subplots(2, 2)
    plot_score_sum(axs[0, 0], moba_without_liar, 'upper left')
    plot_score_sum(axs[0, 1], moba_with_liar, 'upper left')


    axs[0, 0].set_title('(1) Without liar', fontdict={'fontsize': 'x-large'})
    axs[0, 1].set_title('(2) With Liar', fontdict={'fontsize': 'x-large'})
    axs[0, 0].set_ylim(-0.5, 16.5)
    axs[0, 1].set_ylim(-0.5, 16.5)
    plt.suptitle('(a) Sum of the dubious and validity scores - Game1 (N vs N)', fontsize='xx-large', fontweight='bold')

    plot_score_sum(axs[1, 0], action_without_liar, 'lower right')
    plot_score_sum(axs[1, 1], action_with_liar, 'lower right', True)
    axs[1, 0].set_title('(1) Without liar', fontdict={'fontsize': 'x-large'})
    axs[1, 1].set_title('(2) With liar', fontdict={'fontsize': 'x-large'})
    plt.figtext(0.5, 0.485, "(b) Sum of the dubious and validity scores - Game2 (1 vs 1)", ha='center', va='center', fontdict={'fontsize': 'xx-large', 'fontweight': 'bold'})
    axs[1, 0].set_ylim(-20.5, 20.5)
    axs[1, 1].set_ylim(-20.5, 20.5)

      
    fig.set_figwidth(10)
    fig.set_figheight(8)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.savefig(fname='img/figure1.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(fname='img/figure1.png', bbox_inches='tight', pad_inches=0.1)
    
def figure2(user_without_liar, user_with_liar):
    fig, axs = plt.subplots(1, 2)

    plot_voted_count(axs[0], user_without_liar)
    plot_voted_count(axs[1], user_with_liar, True)


    axs[0].set_title('(1) Without liar')
    axs[1].set_title('(2) With liar')
    
    plt.suptitle('Number of matches ended up with certain condition', x=0.5, y=1.05,fontsize='xx-large', fontweight='bold')
    # Adjust vertical_spacing = 0.5 * axes_height
    #plt.subplots_adjust(wspace=0.4, hspace=1)

    # Add text in figure coordinates
    
    fig.set_figwidth(8)
    fig.set_figheight(3.2)
    #plt.subplots_adjust(wspace=0.2, hspace=0.5)

    plt.savefig(fname='img/figure2.pdf', bbox_inches='tight', pad_inches=0.23)  
    plt.savefig(fname='img/figure2.png', bbox_inches='tight', pad_inches=0.23)  
    