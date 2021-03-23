#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import asyncio
import json
import websockets
#import redis
import time
from time import gmtime, strftime
import socket
import concurrent.futures
import sqlalchemy as sa
import asyncpg
{'table': 'tradeBin1m', 'action': 'partial', 'keys': [], 'types': {'timestamp': 'timestamp', 'symbol': 'symbol', 'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'trades': 'long', 'volume': 'long', 'vwap': 'float', 'lastSize': 'long', 'turnover': 'long', 'homeNotional': 'float', 'foreignNotional': 'float'}, 'foreignKeys': {'symbol': 'instrument'}, 'attributes': {'timestamp': 'sorted', 'symbol': 'grouped'}, 'filter': {'symbol': 'XBTUSD'}, 'data': [{'timestamp': '2020-11-08T03:58:00.000Z', 'symbol': 'XBTUSD', 'open': 14993.5, 'high': 14993.5, 'low': 14986.5, 'close': 14993.5, 'trades': 84, 'volume': 364183, 'vwap': 14992.5037, 'lastSize': 24, 'turnover': 2429410308, 'homeNotional': 24.29410308, 'foreignNotional': 364183}]}
{'table': 'tradeBin1m', 'action': 'insert', 'data': [{'timestamp': '2020-11-08T03:59:00.000Z', 'symbol': 'XBTUSD', 'open': 14993.5, 'high': 14993.5, 'low': 14987, 'close': 14987.5, 'trades': 67, 'volume': 307919, 'vwap': 14992.5037, 'lastSize': 233, 'turnover': 2053941004, 'homeNotional': 20.539410039999996, 'foreignNotional': 307919}]}

### DERIBIT ######################################
engine = sa.create_engine('postgresql://postgres:T0raja$am@localhost:5433/baitulbayanat')

def check_table(tbl, engine, exchange):
    if not engine.dialect.has_table(engine, tbl):
        if exchange == "derbit" :
            metadata = sa.MetaData(engine)
            sa.Table(tbl, metadata,
                     sa.Column('time', sa.String),
                     sa.Column('open', sa.Float()),
                     sa.Column('close', sa.Float()),
                     sa.Column('high', sa.Float()),
                     sa.Column('low', sa.Float()),
                     sa.Column('volume', sa.Float()))
            metadata.create_all()
        elif exchange == "bitmex" :
            sa.Table(tbl, metadata,
                     sa.Column('time', sa.String),
                     sa.Column('open', sa.Float()),
                     sa.Column('close', sa.Float()),
                     sa.Column('high', sa.Float()),
                     sa.Column('low', sa.Float()),
                     sa.Column('volume', sa.Float()),
                     sa.Column('trades', sa.INTEGER()),
                     sa.Column('vwap', sa.Float()),
                     sa.Column('lastSize', sa.Float()))
            metadata.create_all()
class ToDbDeribit:
    def __init__(self, jload):
        self.jload = jload

    def conv(self, tbl, exchange):
        if exchange == "deribit" :
            value = {
                'time': self.jload['params']['data']['tick'],
                'open' : self.jload['params']['data']['open'],
                'close' : self.jload['params']['data']['close'],
                'high' : self.jload['params']['data']['high'],
                'low' : self.jload['params']['data']['low'],
                'volume': self.jload['params']['data']['volume']
            }
        if exchange == "bitmex" :
            value = {
                'time': self.jload['data'][0]['tick'],
                'open' : self.jload['data'][0]['open'],
                'close' : self.jload['data'][0]['close'],
                'high' : self.jload['data'][0]['high'],
                'low' : self.jload['data'][0]['low'],
                'volume': self.jload['data'][0]['volume'],
                'trades': self.jload['data'][0]['trades'],
                'vwap': self.jload['data'][0]['vwap'],
                'lastSize': self.jload['data'][0]['lastSize']
            }
        nm = list(value.keys())
        val = list(value.values())
        z = [str(i) for i in val]
        z[0] = "'" + z[0] + "'"
        z2 = ",".join(z).replace("time", "'time'")
        qr = "INSERT INTO " + tbl + "(" + ','.join(nm) + ") " + "VALUES (" + z2 + ")"
        return qr

def message(channel, exchange) :
    if exchange == "deribit" :
        msg = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "public/subscribe",
            "params": {
                "channels": channel
            }
        }
    elif

class ws():

    def __init__(self, ws):
        self.websocket = websockets.connect(ws)

    def subscr(self, channel, msg):
        self.channel = channel
        msg = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "public/subscribe",
            "params": {
                "channels": channel
            }
        }
        return json.dumps(msg)


    async def pubStream(self, msg, tbl, exchange):
        con = await asyncpg.connect('postgresql://postgres:T0raja$am@localhost:5433/baitulbayanat')
        async with self.websocket as websocket:  # creat the connection
            await websocket.send(msg)  # subcription message send to API
            try:
                while websocket.open:
                    response = await websocket.recv()  # string
                    jload = json.loads(response)  # dict
                    try:
                        query = ToDbDeribit(jload).conv(tbl)
                        print(query)
                        await con.execute(query=query)
                    except Exception as ez:
                        print(ez)
                    # with open('deribit_20200803_2.txt', 'a') as the_file:
                    #    the_file.write(str(jload))
                print('websocket close')

            except (ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.CancelledError,
                    websockets.exceptions.WebSocketException) as e:
                print( exchange + ' Public ' + self.channel  + ' WS down. Restarting:',
                      strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())), str(e))

            except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError,
                    concurrent.futures._base.CancelledError) as e:
                print(exchange + ' Public ' + self.channel  + ' WS down. Restarting:',
                      strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())), str(e))

            except socket.gaierror as err:
                print('socket err' + self.channel  + ':', str(err))

            except OSError as err:
                print("OS error: {0}".format(err))

            except Exception as e:
                print(str(e), self.channel , strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())))


ws_deribit = 'wss://www.deribit.com/ws/api/v2'


channelDbBTCP = ["chart.trades.BTC-PERPETUAL.1"]  # BTC PERP
tbl_deribit = 'derbit_ticker_btc'
msgStreamDbBTCP = deribit(ws_deribit).subscr(channelDbBTCP)

check_table_deribit(tbl_deribit, engine)

try:
    loop = asyncio.get_event_loop()
    loop.create_task(deribit(ws_deribit).pubStream(msgStreamDbBTCP, tbl_deribit))
    loop.run_forever()
except KeyboardInterrupt:
    loop.stop()