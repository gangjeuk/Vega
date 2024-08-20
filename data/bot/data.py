import os, sys
import pandas as pd
from datetime import datetime, timedelta

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir)

dateparse = lambda x: datetime.strptime(x, '%Y%m%d%H%M%S.%f')

botlist = pd.read_csv(os.path.join(file_dir,'bot_1221_0113.csv'), parse_dates=['PT_KEY'], date_format='%Y%m%d%H%M%S.%f')

friend = pd.read_csv(os.path.join(file_dir, r'friend_upto_20110113.csv'))

ONE_DAY = timedelta(days=1)
START_DAY = datetime.strptime('20101221', '%Y%m%d')
END_DAY = datetime.strptime('20110113', '%Y%m%d')


# Cheaters' bot usage days
# return {cheater_id: days}
def __bot_usage_days():
    now_day = START_DAY
    cheater_set = dict()
    while now_day <= END_DAY:
        next_day = now_day + ONE_DAY
        bot_per_day = set(botlist[((now_day <= botlist.PT_KEY) & (botlist.PT_KEY < next_day))].Actor)

        for b in bot_per_day:
            if b not in cheater_set.keys():
                cheater_set[b] = 1
            else:
                cheater_set[b] += 1
        
        now_day = next_day
    return cheater_set

bot_usage_days = __bot_usage_days()

def user_used_bot(start_day, end_day):
    global botlist
    used_bot = set(botlist[((start_day <= botlist.PT_KEY) & (botlist.PT_KEY < end_day))].Actor)
    return used_bot

if __name__ == "__main__":

    # 치팅 인원의 변화
    cheater_set = set()
    friend_network = set(friend.actor) | set(friend.target)
    start_day = START_DAY
    while start_day <= END_DAY:
        next_day = start_day + ONE_DAY
        bot_per_day = set(botlist[((start_day <= botlist.PT_KEY) & (botlist.PT_KEY < next_day))].Actor)
        
        print("Total cheater: ", len(bot_per_day))
        print("Bot user in friend network: ", len(bot_per_day & friend_network))
        print("New cheater in friend network: ", len((bot_per_day & friend_network) - cheater_set))
        print("Infection total: ", len(cheater_set))
        cheater_set = cheater_set | (bot_per_day & friend_network)
        start_day = next_day
        
