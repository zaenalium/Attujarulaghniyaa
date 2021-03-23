import json
import aiohttp
import asyncio
import sqlalchemy as sa
# import os
# from aiohttp import web
from datetime import datetime
import asyncpg
engine = sa.create_engine('postgresql://postgres:T0raja$am@localhost:5433/baitulbayanat')

def conv(jload, tbl):
    value = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
        'response': json.dumps(jload)
    }
    nm = list(value.keys())
    val = list(value.values())
    z = [str(i) for i in val]
    z[0] = "'" + z[0] + "'"
    z[1] = "'" + z[1] + "'"
    z2 = ",".join(z).replace("time", "'time'")
    qr = "INSERT INTO " + tbl + "(" + ','.join(nm) + ") " + "VALUES (" + z2 + ")"
    return qr

def check_table(tbl, engine, truncate_table = False):
    if truncate_table == True :
        con2 = engine.connect()
        con2.execute('truncate table '+tbl)
        con2.close()
    if not engine.dialect.has_table(engine, tbl):
        metadata = sa.MetaData(engine)
        sa.Table(tbl, metadata,
                 sa.Column('time', sa.String),
                 sa.Column('response', sa.JSON()))
                 #sa.Column('sell', sa.Float()))
        metadata.create_all()


async def get_rest_indodax(method,symb, tbl):
    con = await asyncpg.connect('postgresql://postgres:T0raja$am@localhost:5433/baitulbayanat')
    try :
        while True:
            url = 'https://indodax.com/api/' + method+ '/' + symb
            async with aiohttp.ClientSession() as session:
                response = await session.get(url=url)
                html = await response.text()
                html = str(html).replace('"', '').replace("buy", "'buy'").replace("sell", "'sell'")
                htmlx = eval(html)
                query = conv(htmlx, tbl)
                await con.execute(query=query)
                print(query)
                print("waiting for 2 second")
                await asyncio.sleep(2)
    except Exception as e :
        print(e)


check_table('indodax_depth_sushi_idr', engine)
check_table('indodax_depth_waves_idr', engine)
check_table('indodax_depth_okb_idr', engine)
check_table('indodax_depth_vidy_idr', engine)

loop = asyncio.get_event_loop()
try:
    asyncio.ensure_future(get_rest_indodax('depth','sushiidr', 'indodax_depth_sushi_idr'))
    asyncio.ensure_future(get_rest_indodax('depth', 'wavesidr', 'indodax_depth_waves_idr'))
    asyncio.ensure_future(get_rest_indodax('depth', 'okbidr', 'indodax_depth_okb_idr'))
    asyncio.ensure_future(get_rest_indodax('depth','vidyidr', 'indodax_depth_vidy_idr'))
    loop.run_forever()
except KeyboardInterrupt:
    pass
finally:
    print("Closing Loop")
    loop.close()