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


### BITSTAMP ##########################################################################################

class okex():

    def __init__(self):
        self.websocket = websockets.connect("wss://real.okex.com:8443/ws/v3")

    def subscr(self, symb):
        msg = {
            "op": "subscribe",
            "args": symb
            }
        return json.dumps(msg)

    async def pubStream(self, msg):

        async with self.websocket as websocket:  # create the connexion
            await websocket.send(msg)  # send the message

            try:
                while websocket.open:
                    response = await websocket.recv()  # string,  receive the response
                    #jload = json.loads(response)  # dict
                    print(response)

            except (ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.CancelledError,
                    websockets.exceptions.WebSocketException) as e:
                print('okex Public WS down. Restarting : ', str(e))

            except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError,
                    concurrent.futures._base.CancelledError) as e:
                print('okex Public WS down. Restarting : ', str(e))

            except socket.gaierror as err:
                print('socket err:', str(err))

            except OSError as err:
                print("OS error: {0}".format(err))


#             except Exception as e:
#                 print(str(e), symbol, strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())))

chn = ["spot/ticker:ETH-USDT", "spot/candle60s:ETH-USDT"]
msgStreamBitmex = okex().subscr(chn)

try:
    loop = asyncio.get_event_loop()
    loop.create_task(okex().pubStream(msgStreamBitmex))
    loop.run_forever()
except KeyboardInterrupt:
    loop.stop()