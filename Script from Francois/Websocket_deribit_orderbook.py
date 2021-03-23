#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import asyncio
import json
import websockets
import redis
import time
from time import gmtime, strftime
import socket
import concurrent.futures

### DERIBIT ######################################

class deribit():

    def __init__(self):
        self.websocket = websockets.connect('wss://www.deribit.com/ws/api/v2')
        
        
    def subscr(self, channel):
        msg = {
                "jsonrpc" : "2.0",
                "id" : 42,
                "method" : "public/subscribe",
                "params" : {
                            "channels" : channel
                            }
                }
        return json.dumps(msg)
    
    async def pubStream(self, msg,symbol):
        
        async with self.websocket as websocket: # creat the connection
            await websocket.send(msg) # subcription message send to API 
            
            try:
                while websocket.open :
                    response = await websocket.recv() # string
                    jload = json.loads(response) # dict
                    print(jload) # data from the API
                    
                print('websocket close')
        
            except (ConnectionError,asyncio.TimeoutError,asyncio.IncompleteReadError,asyncio.CancelledError,websockets.exceptions.WebSocketException) as e:
                print('DERIBIT Public '+symbol+' WS down. Restarting:', strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())), str(e))

            
            except (concurrent.futures.CancelledError,concurrent.futures.TimeoutError,concurrent.futures._base.CancelledError) as e:
                print('DERIBIT Public '+symbol+' WS down. Restarting:', strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())), str(e))
                
            except socket.gaierror as err:
                print('socket err' +symbol+':', str(err))
                
            except OSError as err:
                print("OS error: {0}".format(err))

            except Exception as e:
                print(str(e), symbol, strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())))
                


# In[ ]:


product = 'BTC-PERPETUAL' # choose the futur you want
depth = '10' # number of order you want between 1, 10 and 20 
channelDbBTCP = ["book."+product+".none."+depth+".100ms"] # BTC PERP
msgStreamDbBTCP = deribit().subscr(channelDbBTCP)

try:
    loop = asyncio.get_event_loop()
    loop.create_task(deribit().pubStream(msgStreamDbBTCP,product))
    loop.run_forever()
except KeyboardInterrupt :
    loop.stop()


# In[ ]:




