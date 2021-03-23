# -*- coding: utf-8 -*-
"""

@author: FQ
"""
from time import time

def IdTradeStrings(df, dir):

    #df = df.iloc[515:550]
    prev_bidL1 = 2241900

    t_thresh = 60
    prc_push_thresh = 10000
    prc_fall_thresh = .8
    
    t_elapse = 0
    prc_push_tot = 0
    prc_cancfall_tot = 0
    start_flg = True
    pushing_flg = True
    falling_flg = False
    pattern_flg = False
    pattern_cnt = 0
    
    
    for row in df.itertuples():
        i = row[0]
        print ("i:", i)
        
        if start_flg:    #start checking for tradestring
            t_start = row.time
            start_flg = False
        
        # a buy order pushes L1 up
        #if dir == 1 and row.type == 1 and prev_bidL1 < row.bid_prc_L1:
        
        prc_push = row.bid_prc_L1 - prev_bidL1
        prc_push_tot += prc_push

        print ("ord type / price", row.type, row.price)
        print ("prc_push --> prc_push_tot", prc_push, "-->", prc_push_tot)

        if prc_push_tot >= prc_push_thresh:
            falling_flg = True
            print ("price push thresh reached")
        
        if falling_flg:
            # a buy order cancel or delete causes L1 to fall (down)
            if dir == 1 and (row.type == 2 or row.type == 3) and prev_bidL1 > row.bid_prc_L1:
                prc_cancfall =  prev_bidL1 - row.bid_prc_L1
                prc_cancfall_tot += prc_cancfall
                print ("this is part of fall")
                
                if prc_cancfall_tot >= prc_fall_thresh * prc_push_thresh:
                    pattern_flg = True
                    pattern_cnt += 1
                    print ("#{} Pattern Found: push{} -- start {} -- end {}".format(pattern_cnt, prc_push_tot, t_start, row.time))
                    
                    
        t_elapse = row.time - t_start
        
        if pattern_flg == True:
            print (i, row.time, t_elapse, row.type, row.price, "-----\n")
        
        if t_elapse >= t_thresh or pattern_flg == True:
            t_elapse = 0
            prc_push_tot = 0
            prc_cancfall_tot = 0
            start_flg = True
            pattern_flg = False
    
        prev_bidL1 = row.bid_prc_L1
        
    
def IdTradeStrings2(df, fn_start=-1, fn_end=-1):
    debug = True
    flog = open('IDTradeStringsLog.txt', 'w')
    
    t_thresh = 1 
    prc_push_thresh = 1000
    prc_fall_thresh = .8
    
    pushed_flg = False
    fallen_flg = False
    
    if (fn_start==-1):
        fn_start = df['time'].min()
    if (fn_end==-1):
        fn_end = df['time'].max()
    
    #use these functions to convert time to index
    fn_start_idx = df[(df['time'] > fn_start)]['time'].idxmin()
    fn_end_idx = df[(df['time'] < fn_end)]['time'].idxmax()
    
    str_start_idx = fn_start_idx
    str_end_idx = df[(df['time'] > fn_start + t_thresh)]['time'].idxmin()
    
    str_start = df.loc[str_start_idx, 'time']
    str_end = df.loc[str_end_idx, 'time']
    
    if debug:
        flog.write("Function Start: [{}] {}\n".format(str_start_idx, str_start))
        flog.write("Function End: [{}] {}\n".format(fn_end_idx, df.loc[fn_end_idx, 'time']))
    
    while str_end_idx <= fn_end_idx:
        
        string = df.loc[str_start_idx:str_end_idx]
        
        if debug:
            flog.write ("--------\n")
            flog.write(repr(string.iloc[:,[0,8]])+"\n")
            flog.write ("Begin Loop on Interval [{}] {} [{}] {}\n".format(str_start_idx, str_start, str_end_idx, str_end))
            
        #check string for some bad conditions
        #string should not be empty, have a single row, or have a single time
        empty_flg = string.empty
        one_row_flg = (string.shape[0]==1)
        one_time_flg = False
        
        if not empty_flg:
            one_time_flg = (string.iloc[0]['time']==string.iloc[-1]['time'])
        
        if debug:
            flog.write ("Check Bad String - Empty:{} Single Row:{} Single Time:{}\n".format(empty_flg, one_row_flg, one_time_flg)
)            
        # if any bad conditions found, don't process this string move to next
        if not (empty_flg or one_row_flg or one_time_flg):
            
            str_start_prc = string.loc[str_start_idx, 'bid_prc_L1']
            peak = string['bid_prc_L1'].max()
            
            # only porcess further if there is a peak
            if peak > str_start_prc:
                   
                # retrieve index of first occurance of maximim price in string
                peak_idx = string['bid_prc_L1'].idxmax()
                peak_time = string.loc[peak_idx,'time']
                #print ("Peak Exists", peak_idx, peak, peak_time)
                if debug: 
                    flog.write ("Peak {} > Start Price {}\n".format(peak, str_start_prc))
                    flog.write ("Peak @ [{}] {}\n".format(peak_idx, peak_time))
                
                # update start to min L1_bid point between original start and peak
                str_start_idx = string.loc[str_start_idx:peak_idx]['bid_prc_L1'].idxmin()
                str_start = string.loc[str_start_idx, 'time']
                str_start_prc = string.loc[str_start_idx, 'bid_prc_L1']
                #str_end_idx = df[(df['time'] < str_start + t_thresh)]['time'].idxmax()
                #str_end = df.loc[str_end_idx, 'time']
                #str_end_prc = df.loc[str_end_idx, 'bid_prc_L1']
                if debug: 
                    flog.write ("Find Min before Peak\n")
                    flog.write ("Updated Start: [{}] {} Start Price:{}\n".format(str_start_idx, str_start, str_start_prc))
                    flog.write ("Updated End: [{}] {} End\n".format(str_end_idx, str_end))

                # peak meets height requirement (i.e. prc_push_thresh)
                if (peak - str_start_prc) > prc_push_thresh:
                    pushed_flg = True
                    if debug: flog.write ("Peak is High Enough\n")

                
                #now check for lower minimum between peak and end 
                #if there is, that minimum, becomes new end
                se = df.loc[peak_idx+1:str_end_idx]
                if not se.empty:
                    #str_end_idx1 = se['bid_prc_L1'].idxmin()
                    # below gets last occurance of min price (we reverese order of se)
                    str_end_idx = se['bid_prc_L1'].iloc[::-1].idxmin()
                    str_end = df.loc[str_end_idx, 'time']
                    str_end_prc = df.loc[str_end_idx, 'bid_prc_L1']
                    if debug:
                        flog.write ("Check Min after Peak\n")
                        flog.write ("Updated End: [{}] {} End Price:{}\n".format(str_end_idx, str_end, str_end_prc))
                
                else:
                    if debug:
                        flog.write ("No gap between peak and end\n")
            
    
                if (peak - str_end_prc) > (peak - str_start_prc) * prc_fall_thresh:
                    fallen_flg = True
                    if debug:
                        flog.write ("Price Fall meets threshold: Fall {} Thresh {} \n".format(peak-str_end_prc, (peak - str_start_prc) * prc_fall_thresh))
            
            if pushed_flg == True and fallen_flg ==True:
                print ("RECORD PATTERN", str_start_idx, str_start, str_end_idx, str_end)
                if debug:
                    flog.write ("RECORD PATTERN [{}] {} [{}] {} \n".format(str_start_idx, str_start, str_end_idx, str_end))
            else:
                if debug:
                    flog.write ("NOT RECORDED [{}] {} [{}] {} \n".format(str_start_idx, str_start, str_end_idx, str_end))
        
        #print ("--------")
        #print ("Completed String", str_start_idx, str_end_idx)
        str_start_idx = str_end_idx+1
        str_start = df.loc[str_start_idx, 'time']
        str_end_idx = df[(df['time'] < str_start + t_thresh)]['time'].idxmax()
        str_end = df.loc[str_end_idx, 'time']
        pushed_flg = False
        fallen_flg = False
    
    flog.close()
    
    
    
    
    
        
    #df_string = df[(df['time'] >= int_start) & (df['time'] < int_end)]