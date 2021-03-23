#wss://api2.btcturk.com


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


### btcturk ##########################################################################################

class btcturk():

    def __init__(self):
        self.websocket = websockets.connect('wss://ws-feed-pro.btcturk.com/')

    def subscr(self, id_symb):
        msg = [151,
                    {
                        'type': 151,
                        'channel': 'ticker',
                        'event': id_symb,
                        'join': True
                    }
                ]
        return json.dumps(msg)

    async def pubStream(self, msg):

        async with self.websocket as websocket:  # create the connexion
            await websocket.send(msg)  # send the message

            try:
                while websocket.open:
                    response = await websocket.recv()  # string,  receive the response
                    jload = json.loads(response)  # dict
                    print(jload)

            except (ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.CancelledError,
                    websockets.exceptions.WebSocketException) as e:
                print('btcturk Public WS down. Restarting : ', str(e))

            except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError,
                    concurrent.futures._base.CancelledError) as e:
                print('btcturk Public WS down. Restarting : ', str(e))

            except socket.gaierror as err:
                print('socket err:', str(err))

            except OSError as err:
                print("OS error: {0}".format(err))

            except Exception as e:
                print(str(e), strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())))

chn = "XRPUSDT"
msgStreamBitmex = btcturk().subscr(chn)

try:
    loop = asyncio.get_event_loop()
    loop.create_task(btcturk().pubStream(msgStreamBitmex))
    loop.run_forever()
except KeyboardInterrupt:
    loop.stop()