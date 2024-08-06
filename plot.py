import os
from ContationDataset.data import START_DAY, END_DAY, ONE_DAY

import matplotlib.pyplot as plt
from matplotlib import colormaps

def plot_score_sum(ax, user, loc):

    benign_user = user.loc[(user.used_bot.eq(False))]
    cheater_user = user.loc[(user.used_bot.eq(True))]
    x = cheater_user.val 
    y = cheater_user.dub 

    color = cheater_user.played_match
    size = cheater_user.groupby(['dub','val'])['id'].transform('count') + 50
    
    
    scatter = ax.scatter(x, y, s= size, c=color, marker='^', cmap="Dark2", label='Cheater')
    ax.legend(loc='upper right')
    
    # Color class for cheater or user
    legend1 = ax.legend(*scatter.legend_elements(prop="colors", num=5),
                        loc=loc, title="Played match")
    ax.add_artist(legend1)

    # Size for number of user with a same value
    handles, labels = scatter.legend_elements(prop="sizes", alpha=1, num = 4)
    legend2 = ax.legend(handles, labels, loc="lower right", title="Sizes")
    
    x = benign_user.val 
    y = benign_user.dub 
    size = benign_user.groupby(['dub','val'])['id'].transform('count') + 50
    scatter = ax.scatter(x, y, s= size, c='b', label='Normal User')
    ax.legend(loc = "upper right")
    
    ax.set_xlabel('Sum of validity')
    ax.set_ylabel('Sum of dubious')
    
    return ax    

def plot_voted_count(ax, user):

    benign_user = user.loc[(user.used_bot.eq(False))]
    cheater_user = user.loc[(user.used_bot.eq(True))]

    
    x = benign_user.voted_to_cheat 
    y = benign_user.voted_to_untrue 
    color = benign_user.bot_usage_days
    size = benign_user.groupby(['dub','val'])['id'].transform('count') + 50
    scatter = ax.scatter(x, y, s= size, c='b', cmap="Spectral", label='Normal User')
    
    x = cheater_user.voted_to_cheat
    y = cheater_user.voted_to_untrue
    color = cheater_user.played_match
    size = cheater_user.groupby(['dub','val'])['id'].transform('count') + 50
    
    
    scatter = ax.scatter(x, y, s= size, c=color, marker='^', cmap='Dark2', label='Cheater')    
    legend1 = ax.legend(*scatter.legend_elements(prop="colors", num=7),
                        loc="lower right", title="Played match")
    ax.add_artist(legend1)
    ax.legend()
    
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_xlabel('Ended up with high dubious (<0.0)')
    ax.set_ylabel('Ended up with low validity (<0.5)')
    
    return ax
    #plt.savefig(os.path.join('img', filename), bbox_inches='tight', pad_inches=0)


def figure1(user_with_liar, user_without_liar):
    fig, axs = plt.subplots(1, 2)
    plot_score_sum(axs[0], user_without_liar, 'upper left')
    plot_score_sum(axs[1], user_with_liar, 'upper left')
    
    axs[0].set_title('(1) Without liar')
    axs[1].set_title('(2) With Liar')
    
    plt.suptitle('Sum of the dubious and validity scores - Game1 (N vs N)', fontsize='x-large', fontweight='bold')
    
    fig.set_figwidth(10)
    fig.set_figheight(4.5)
    
    plt.savefig(fname='img/figure1.png', bbox_inches='tight', pad_inches=0)
    
def figure2(user_with_liar, user_withou_liar):
    fig, axs = plt.subplots(2, 2)
    plot_score_sum(axs[0, 0], user_withou_liar, 'lower left')
    plot_score_sum(axs[0, 1], user_with_liar, 'lower left')
    plot_voted_count(axs[1, 0], user_withou_liar)
    plot_voted_count(axs[1, 1], user_with_liar)

    axs[0, 0].set_title('(1) Without liar')
    axs[0, 1].set_title('(2) With liar')
    axs[1, 0].set_title('(1) Without liar')
    axs[1, 1].set_title('(2) With liar')
    
    plt.suptitle("Sum of the dubious and validity scores - Game2 (1 vs 1)", fontsize='x-large', fontweight='bold')
    # Adjust vertical_spacing = 0.5 * axes_height
    plt.subplots_adjust(hspace=0.5)

    # Add text in figure coordinates
    plt.figtext(0.5, 0.485, 'Number of matches ended up with certain condition', ha='center', va='center', fontdict={'fontsize': 'x-large', 'fontweight': 'bold'})
    
    fig.set_figwidth(12)
    fig.set_figheight(9)
    plt.savefig(fname='img/figure2.png', bbox_inches='tight', pad_inches=0)  
    