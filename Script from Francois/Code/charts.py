# -*- coding: utf-8 -*-
"""

@author: FQ
"""
import math
3


def print_chart(df, tslice, start_time=-1, end_time=-1):
        
    df_trades = df[(df['type'] == 4) | (df['type'] == 5)]
    df_buy_cancel = df[(df['dir']==1) & ((df['type'] == 2) | (df['type'] == 3))]
    df_buy_create = df[(df['dir']==1) & (df['type'] == 1)]
    df_sell_cancel = df[(df['dir']==-1) & ((df['type'] == 2) | (df['type'] == 3))]
    df_sell_create = df[(df['dir']==-1) & (df['type'] == 1)]
    
    if start_time == -1 and end_time == -1:
        start_time = df['time'].min()
        end_time = df['time'].max()
            
    all_time = end_time - start_time 
       
    remainder, n_ints = math.modf(all_time / tslice)
    #start_time = start_time + tslice*0.5*remainder
    #end_time = end_time - tslice*0.5*remainder
    
    n_ints += 1   # run a final iteration even if there are no datapoints for entire time slice
    int_start = start_time
    int_end = int_start + tslice
    
    for t in range (int(n_ints)):
        
        df_sl = df[(df['time'] >= int_start) & (df['time'] < int_end)]
        if not df_sl.empty:
            
            '''
            Plot horizontal line chart for the L1 Order Book Price Levels
            df_sl contains all data rows (and all columns) for the current time slice
            df_sl1 will contain "end times"
            The horizonal lines should be plotted at
            y=df_sl.bid/ask_prc_L1 from x1=df_sl.time to x2 = df_sl1.time
            '''
                  
            sl_1st_ind = df_sl.first_valid_index()
            sl_last_ind = df_sl.last_valid_index() 
            df_sl1 = df.loc[sl_1st_ind+1:sl_last_ind+1,['time', 'bid_prc_L1', 'ask_prc_L1']]
        
            plt.figure(figsize=(10,5))
             
            plt.hlines (df_sl['bid_prc_L1'], df_sl['time'], df_sl1['time'], colors='b')
            plt.hlines (df_sl['ask_prc_L1'], df_sl['time'], df_sl1['time'], colors='r')
            #plt.vlines (df_sl['time'], df_sl['bid_prc_L1'], df_sl1['bid_prc_L1'], colors='b')
            #plt.vlines (df_sl['ask_prc_L1'], df_sl['time'], df_sl1['time'], colors='r')
        
            '''
            Now plot Creates and Cancels
            '''
    
            df_trades_sl = df_trades[(df_trades['time'] >= int_start) & (df_trades['time'] < int_end)]
            df_buy_create_sl = df_buy_create[(df_buy_create['time'] >= int_start) & (df_buy_create['time'] < int_end)]
            df_buy_cancel_sl = df_buy_cancel[(df_buy_cancel['time'] >= int_start) & (df_buy_cancel['time'] < int_end)]
            df_sell_create_sl = df_sell_create[(df_sell_create['time'] >= int_start) & (df_sell_create['time'] < int_end)]
            df_sell_cancel_sl = df_sell_cancel[(df_sell_cancel['time'] >= int_start) & (df_sell_cancel['time'] < int_end)]
        
            plt.plot (df_trades_sl['time'], df_trades_sl['price'], 'ko')
        
            plt.plot (df_buy_create_sl['time'], df_buy_create_sl['price'], 'b+')
            plt.plot (df_buy_cancel_sl['time'], df_buy_cancel_sl['price'], 'bx')
       
            plt.plot (df_sell_create_sl['time'], df_sell_create_sl['price'], 'r+')
            plt.plot (df_sell_cancel_sl['time'], df_sell_cancel_sl['price'], 'rx')
            
            plt.title("Chart #{0} Start:{1} End:{2}".format(t, int_start, int_end)) 
            ax=plt.gca()
            ax.get_xaxis().get_major_formatter().set_useOffset(False)
            plt.show()
        
        int_start = int_start + tslice
        int_end = int_start + tslice
        #if t == 2:
        #    break