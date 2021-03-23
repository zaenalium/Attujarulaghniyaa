#!/usr/bin/env python
# coding: utf-8
import asyncio
import json
import websockets
import time
import redis
import time
from time import gmtime, strftime
import socket
import concurrent.futures
### CRYPTOFACILITIES ################################################################################
class cryptofacilities():
    def __init__(self):
        self.websocket = websockets.connect('wss://www.cryptofacilities.com/ws/v1')
    def subscr(self, feed, symbol):
        msg = {
            "event":"subscribe",
            "feed":feed,
            "product_ids":symbol
        }
        return json.dumps(msg)
    
    async def pubStream(self, msg):
        async with self.websocket as websocket:
            await websocket.send(msg)
            try:
                while websocket.open :
                    response = await websocket.recv() # string
                    jload = json.loads(response) # dict
                    with open('cryptofacilities_btc_20200822_new.txt', 'a') as the_file:
                        the_file.write(str(jload) + "\n")
                    
                print('websocket close')
            except Exception as e:
                print(str(e), symbol, strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())))
feedCf = "book" # to get orderbook
symbolsCf = ["pi_xbtusd","fi_xbtusd_200925","pi_ethusd","fi_ethusd_200925"] # list of products the channels subscribe 
# "pi_xbtusd" and "pi_ethusd" are perpetual BTC and ETH
# "fi_xbtusd_200925" and "fi_ethusd_200925" are futures 25 september BTC and ETH 
# you can choose only one, or different, put off the list the one you don't want
msgFeedCf = cryptofacilities().subscr(feedCf, symbolsCf)
try:
    loop = asyncio.get_event_loop()
    loop.create_task(cryptofacilities().pubStream(msgFeedCf))
    loop.run_forever()
except KeyboardInterrupt :
    loop.stop()
