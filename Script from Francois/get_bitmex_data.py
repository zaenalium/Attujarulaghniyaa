#!/usr/bin/env python
# coding: utf-8

import asyncio
import json
import websockets
import datetime
import redis
import time
from time import gmtime, strftime
import socket 
import concurrent.futures
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://postgres:T0raja$am@localhost:5432/jarvis')

### BITMEX ##########################################################################################
def replace_quotes(x) :
    x = x.replace("\'", "\"")
    return x
class bitmex():

    def __init__(self):
        self.websocket = websockets.connect('wss://www.bitmex.com/realtime')
        

    def subscr(self, channel):
        msg = {
            "op" : "subscribe",
            "args" : channel
        }
        return json.dumps(msg)
    
    async def pubStream(self, msg):

        async with self.websocket as websocket: # create the connexion 
            await websocket.send(msg) # send the message
            
            try:
                while websocket.open :
                    response = await websocket.recv() # string
                    jload = json.loads(response) # dict
                    try :
                        ts = pd.to_datetime(jload['data'][0]['timestamp'])
                        jload['data'][0].pop('symbol', None)
                        #print('aman #1')
                        df = pd.DataFrame({'time' : [ts],
                                             'data' : [json.dumps(jload['data'][0])]})
                        #print('aman #2')
                        #print(df)
                        df.to_sql("bitmex_json", engine, if_exists='append', chunksize=1000, index=False)
                    except :#OSError as err:
                        print('error cuy')
                        #print("OS error: {0}".format(err))
                        pass
                    #print(jload) # data from the APIindent=2
            except :
                pass
                
#             except (ConnectionError,asyncio.TimeoutError,asyncio.IncompleteReadError,asyncio.CancelledError,websockets.exceptions.WebSocketException) as e:
#                 print('BITMEX Public WS down. Restarting : ', str(e))
            
#             except (concurrent.futures.CancelledError,concurrent.futures.TimeoutError,concurrent.futures._base.CancelledError) as e:
#                 print('BITMEX Public WS down. Restarting : ',str(e))
                
#             except socket.gaierror as err:
#                 print('socket err:', str(err))
                
#             except OSError as err:
#                 print("OS error: {0}".format(err))
                
#             except Exception as e:
#                 print(str(e), symbol, strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())))

topicBit = ["orderBook10:XBTUSD"] 
msgStreamBitmex = bitmex().subscr(topicBit)


loop = asyncio.get_event_loop()
loop.create_task(bitmex().pubStream(msgStreamBitmex))
loop.run_forever()

