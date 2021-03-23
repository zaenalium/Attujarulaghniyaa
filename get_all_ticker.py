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
from datetime import datetime
from random import randint

### DERIBIT ######################################
engine = sa.create_engine('postgresql://postgres:T0raja$am@localhost:5433/baitulbayanat')

def check_table(tbl, engine, exchange, truncate_table = False):
    if truncate_table == True :
        con2 = engine.connect()
        con2.execute('truncate table '+tbl)
        con2.close()
    if not engine.dialect.has_table(engine, tbl):
        metadata = sa.MetaData(engine)
        if exchange == "deribit" :
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
                     sa.Column('lastsize', sa.Float()))
            metadata.create_all()
        elif exchange == "bitfinex" :
            sa.Table(tbl, metadata,
                     sa.Column('time', sa.String),
                     sa.Column('bid', sa.Float()),
                     sa.Column('bid_size', sa.Float()),
                     sa.Column('ask', sa.Float()),
                     sa.Column('ask_size', sa.Float()),
                     sa.Column('daily_change', sa.Float()),
                     sa.Column('daily_change_relative', sa.INTEGER()),
                     sa.Column('last_price', sa.Float()),
                     sa.Column('volume', sa.Float()),
                     sa.Column('high', sa.Float()),
                     sa.Column('low', sa.Float()))
            metadata.create_all()
        elif exchange == "bitstamp" :
            sa.Table(tbl, metadata,
                     sa.Column('time', sa.String),
                     sa.Column('price', sa.Float()),
                     sa.Column('amount', sa.Float()),
                     sa.Column('type', sa.INTEGER()))
            metadata.create_all()
        elif exchange == "ftx" :
            sa.Table(tbl, metadata,
                     sa.Column('time', sa.String),
                     sa.Column('bid', sa.Float()),
                     sa.Column('bid_size', sa.Float()),
                     sa.Column('ask', sa.Float()),
                     sa.Column('ask_size', sa.Float()),
                     sa.Column('last', sa.Float()))
            metadata.create_all()
        elif exchange == "kraken" :
            sa.Table(tbl, metadata,
                     sa.Column('time', sa.String),
                     sa.Column('open', sa.Float()),
                     sa.Column('high', sa.Float()),
                     sa.Column('low', sa.Float()),
                     sa.Column('close', sa.Float()),
                     sa.Column('vwap', sa.Float()),
                     sa.Column('volume', sa.Float()),
                     sa.Column('count', sa.Float()))
            metadata.create_all()
        elif exchange == "coinbase" :
            sa.Table(tbl, metadata,
                     sa.Column('time', sa.String),
                     sa.Column('price', sa.Float()),
                     sa.Column('open_24h', sa.Float()),
                     sa.Column('volume_24h', sa.Float()),
                     sa.Column('low_24h', sa.Float()),
                     sa.Column('high_24h', sa.Float()),
                     sa.Column('volume_30d', sa.Float()),
                     sa.Column('best_bid', sa.Float()),
                     sa.Column('best_ask', sa.Float()))
            metadata.create_all()
        if exchange == "indodax" :
            sa.Table(tbl, metadata,
                     sa.Column('time', sa.String),
                     sa.Column('open', sa.Float()),
                     sa.Column('high', sa.Float()),
                     sa.Column('low', sa.Float()),
                     sa.Column('close', sa.Float()),
                     sa.Column('volume', sa.Float()))
            metadata.create_all()


class ToDb:
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
        elif exchange == "bitmex" :
            value = {
                'time': self.jload['data'][0]['timestamp'],
                'open' : self.jload['data'][0]['open'],
                'close' : self.jload['data'][0]['close'],
                'high' : self.jload['data'][0]['high'],
                'low' : self.jload['data'][0]['low'],
                'volume': self.jload['data'][0]['volume'],
                'trades': self.jload['data'][0]['trades'],
                'vwap': self.jload['data'][0]['vwap'],
                'lastsize': self.jload['data'][0]['lastSize']
            }
        elif exchange == "bitfinex" :
            value = {
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                'bid': self.jload[1][0],
                'bid_size': self.jload[1][1],
                'ask': self.jload[1][2],
                'ask_size': self.jload[1][3],
                'daily_change': self.jload[1][4],
                'daily_change_relative': self.jload[1][5],
                'last_price': self.jload[1][6],
                'volume': self.jload[1][7],
                'high': self.jload[1][8],
                'low': self.jload[1][9] }
        elif exchange == "bitstamp" :
            value = {
                'time': self.jload['data']['timestamp'],
                'price': self.jload['data']['price'],
                'amount': self.jload['data']['amount'],
                'type': self.jload['data']['type']
            }
        elif exchange == "ftx" :
            value = {
                'time': str(self.jload['data']['time']),
                'bid': self.jload['data']['bid'],
                'bid_size': self.jload['data']['bidSize'],
                'ask': self.jload['data']['ask'],
                'ask_size': self.jload['data']['askSize'],
                'last' :  self.jload['data']['last']
            }
        elif exchange == "kraken" :
            value = {
                'time': self.jload[1][1],
                'open' : float(self.jload[1][2]),
                'high' : float(self.jload[1][3]),
                'low' : float(self.jload[1][4]),
                'close' : float(self.jload[1][5]),
                'vwap': float(self.jload[1][6]),
                'volume': float(self.jload[1][7]),
                'count': float(self.jload[1][8])
            }
        elif exchange == "coinbase" :
            value = {
                'time': self.jload['time'],
                'price': self.jload['price'],
                'open_24h' : self.jload['open_24h'],
                'volume_24h' : self.jload['volume_24h'],
                'low_24h' : self.jload['low_24h'],
                'high_24h' : self.jload['high_24h'],
                'volume_30d': self.jload['volume_30d'],
                'best_bid': self.jload['best_bid'],
                'best_ask': self.jload['best_ask']
            }
        elif exchange == "indodax" :
            value = {
                'time': self.jload['tick']['t'],
                'open': self.jload['tick']['o'],
                'high': self.jload['tick']['h'],
                'low': self.jload['tick']['l'],
                'close': self.jload['tick']['c'],
                'volume' : self.jload['tick']['v'],
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
    elif exchange == "bitmex" :
        msg = {
            "op" : "subscribe",
            "args" : channel
        }
    elif exchange == "bitfinex" :
        msg = {
              "event": "subscribe",
              "channel": "ticker",
              "symbol": channel
            }
    elif exchange == "bitstamp" :
        msg = {
                "event": "bts:subscribe",
                "data": {
                    "channel": channel
                }
            }
    elif exchange == "ftx" :
        msg = {
                'op': 'subscribe', 'channel': 'ticker', 'market': channel
            }
    elif exchange == "kraken" :
        msg = {
            "event": "subscribe",
                "pair": channel,
            "subscription": {
                "name": 'ohlc'
            }
        }
    elif exchange == "coinbase" :
        msg = {
            "type": "subscribe",
            "channels": [{"name": "ticker", "product_ids": [channel] }]
        }
    elif exchange == "indodax" :
        msg = {
            "sub": channel,
            "id": str(randint(1, 10))
        }
    return msg


class Ws():

    def __init__(self, exchange):
        self.exchange = exchange
        if exchange == "deribit" :
            ws = 'wss://www.deribit.com/ws/api/v2'
        elif exchange == "bitmex" :
            ws = 'wss://www.bitmex.com/realtime'
        elif exchange == "bitfinex" :
            ws = 'wss://api.bitfinex.com/ws/2'
        elif exchange == "bitstamp" :
            ws = 'wss://ws.bitstamp.net'
        elif exchange == "ftx" :
            ws = 'wss://ftx.com/ws/'
        elif exchange == "kraken" :
            ws = 'wss://ws.kraken.com'
        elif exchange == "coinbase" :
            ws = 'wss://ws-feed.pro.coinbase.com'
        elif exchange == "indodax" :
            ws = 'wss://kline.indodax.com/ws/'
        self.websocket = websockets.connect(ws)

    def subscr(self, channel):
        self.channel = channel
        msg = message(channel, self.exchange)
        return json.dumps(msg)

    async def pubStream(self, msg, tbl):
        con = await asyncpg.connect('postgresql://postgres:T0raja$am@localhost:5433/baitulbayanat')
        async with self.websocket as websocket:  # creat the connection
            await websocket.send(msg)  # subcription message send to API
            try:
                while websocket.open:
                    response = await websocket.recv()  # string
                    jload = json.loads(response)  # dict
                    print(jload)
                    try:
                        query = ToDb(jload).conv(tbl, self.exchange)
                        print(query)
                        await con.execute(query=query)
                    except Exception as ez:
                        print(ez)
                    # with open('deribit_20200803_2.txt', 'a') as the_file:
                    #    the_file.write(str(jload))
                print('websocket close')

            except (ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.CancelledError,
                    websockets.exceptions.WebSocketException) as e:
                print(self.exchange + ' Public '  + ' WS down. Restarting:',
                      strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())), str(e))

            except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError,
                    concurrent.futures._base.CancelledError) as e:
                print(self.exchange + ' Public '  + ' WS down. Restarting:',
                      strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())), str(e))

            except socket.gaierror as err:
                print('socket err' + self.exchange + ':', str(err))

            except OSError as err:
                print("OS error: {0}".format(err))

            except Exception as e:
                print(str(e), self.exchange, strftime("%Y-%m-%d_%H:%M:%S", gmtime(time.time())))

# channel_deribit = ["chart.trades.BTC-PERPETUAL.1"]  # BTC PERP
# channel_bitmex = ["tradeBin1m:XBTUSD"]
#channel_deribit_eth = ["chart.trades.ETH-PERPETUAL.1"]  # BTC PERP
# channel_bitmex_eth = ["tradeBin1m:ETHUSD"]
# channel_bitmex_xrp = ["tradeBin1m:XRPUSD"]
# channel_bitmex_ltc = ["tradeBin1m:LTCUSD"]
# channel_bitfinex = 'tBTCUSD'
#channel_bitstamp = 'live_trades_btcusd'
#channel_ftx = "BTC-PERP"
#channel_kraken = ['BTC/USD']
#channel_coinbase = "BTC-USD"
channel_indodax = "btcidr.kline.1m"



# tbl_deribit = 'deribit_ticker_btc'
# tbl_bitmex = 'bitmex_ticker_btc'
# tbl_deribit_eth = 'deribit_ticker_eth'
# tbl_bitmex_eth = 'bitmex_ticker_eth'
# tbl_bitmex_xrp = 'bitmex_ticker_xrp'
# tbl_bitmex_ltc = 'bitmex_ticker_ltc'
#tbl_bitfinex_btc =  'bitfinex_ticker_btc'
#tbl_kraken_btc = 'kraken_ticker_btc'
tbl_indodax_btc = 'indodax_ticker_btc'


# msgStreamDbBTCP = Ws('deribit').subscr(channel_deribit)
# msgStreamBitmexBTC = Ws('bitmex').subscr(channel_bitmex)
# msgStreamDbeth = Ws('deribit').subscr(channel_deribit_eth)
# msgStreamBitmexeth = Ws('bitmex').subscr(channel_bitmex_eth)
# msg_bmx_xrp = Ws('bitmex').subscr(channel_bitmex_xrp)
# msg_bmx_ltc = Ws('bitmex').subscr(channel_bitmex_ltc)
#msg_bitfinex_btc = Ws('bitfinex').subscr(channel_bitfinex)
msg_indodax_btc = Ws('indodax').subscr(channel_indodax)

# check_table(tbl_deribit, engine, 'deribit')
# check_table(tbl_bitmex, engine, 'bitmex')
#check_table(tbl_bitfinex_btc, engine, 'bitfinex')
check_table(tbl_indodax_btc, engine, 'indodax')


# check_table(tbl_deribit_eth, engine, 'deribit')
# check_table(tbl_bitmex_eth, engine, 'bitmex')
# check_table(tbl_bitmex_xrp, engine, 'bitmex')
# check_table(tbl_bitmex_ltc, engine, 'bitmex')

try:
    loop = asyncio.get_event_loop()
    loop.create_task(Ws('indodax').pubStream(msg_indodax_btc, tbl_indodax_btc))
    loop.run_forever()
except KeyboardInterrupt:
    loop.stop()