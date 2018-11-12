#!/usr/bin/env python
import json
import base64
import hmac
import hashlib
import gzip
import os
import re
import time
import datetime
from threading import Thread
from websocket import create_connection, WebSocketConnectionClosedException, WebSocketTimeoutException
import sys
from itertools import count
from lz4 import frame as lz4f
import argparse

pairList = ["BTC-USD", "ETH-USD", "LTC-USD", "BTC-EUR", "ETH-EUR", "ETH-BTC", "LTC-EUR", "LTC-BTC", "BTC-GBP"]


class NoData(Exception):
    pass


NoDataException = NoData("Data flow stopped")


class WebsocketClient(object):
    def __init__(self, url="wss://ws-feed.gdax.com", products=None, message_type="subscribe", mongo_collection=None,
                 should_print=True, auth=False, api_key="", api_secret="", api_passphrase="",
                 channels=["level2", "heartbeat", "full"]):

        self.url = url
        self.products = products
        self.channels = channels
        self.type = message_type
        self.stop = False
        self.error = None
        self.ws = None
        self.thread = None
        self.auth = auth
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.should_print = should_print
        self.mongo_collection = mongo_collection

    def start(self):
        def _go():
            self._connect()
            self._listen()
            self._disconnect()

        self.stop = False
        self.on_open()
        self.thread = Thread(target=_go)
        self.thread.start()

    def _connect(self):
        if self.products is None:
            self.products = ["BTC-USD"]
        elif not isinstance(self.products, list):
            self.products = [self.products]

        if self.url[-1] == "/":
            self.url = self.url[:-1]

        if self.channels is None:
            sub_params = {'type': 'subscribe', 'product_ids': self.products}
        else:
            sub_params = {'type': 'subscribe', 'product_ids': self.products, 'channels': self.channels}

        if self.auth:
            timestamp = str(time.time())
            message = timestamp + 'GET' + '/users/self'
            message = message.encode('ascii')
            hmac_key = base64.b64decode(self.api_secret)
            signature = hmac.new(hmac_key, message, hashlib.sha256)
            signature_b64 = base64.b64encode(signature.digest())
            sub_params['signature'] = signature_b64
            sub_params['key'] = self.api_key
            sub_params['passphrase'] = self.api_passphrase
            sub_params['timestamp'] = timestamp

        self.ws = create_connection(self.url)
        self.ws.send(json.dumps(sub_params))

        if self.type == "heartbeat":
            sub_params = {"type": "heartbeat", "on": True}
        else:
            sub_params = {"type": "heartbeat", "on": False}
        self.ws.send(json.dumps(sub_params))

    def _listen(self):
        while not self.stop:
            try:
                if int(time.time() % 30) == 0:
                    self.ws.ping("keepalive")
                data = self.ws.recv()
                msg = json.loads(data)
            except ValueError as e:
                self.on_error(e)
            except Exception as e:
                self.on_error(e)
            else:
                self.on_message(msg)

    def _disconnect(self):
        if self.type == "heartbeat":
            self.ws.send(json.dumps({"type": "heartbeat", "on": False}))
        try:
            if self.ws:
                self.ws.close()
        except WebSocketConnectionClosedException as e:
            pass

        self.on_close()

    def close(self):
        self.stop = True
        print(f"Joining thread {self.name}...")
        self.thread.join()

    def on_open(self):
        if self.should_print:
            print("-- Subscribed! --\n")

    def on_close(self):
        if self.should_print:
            print("\n-- Socket Closed --")

    def on_message(self, msg):
        if self.should_print:
            print(msg)
        if self.mongo_collection:  # dump JSON to given mongo collection
            self.mongo_collection.insert_one(msg)

    def on_error(self, e, data=None):
        self.error = e
        self.stop = True
        print('{} - data: {}'.format(e, data))


class MyWebsocketClient(WebsocketClient):
    def __init__(self, name, outfile, count=0, *args, **kwargs):
        super(MyWebsocketClient, self).__init__(*args, **kwargs)
        self.name = name
        self.outfile = outfile
        self.message_count = count

    def on_open(self):
        self.url = "wss://ws-feed.gdax.com/"
        self.products = pairList
        self.type = "heartbeat"

    def on_message(self, msg):
        json.dump(msg, self.outfile)
        self.outfile.write("\n")
        self.message_count += 1

    def on_close(self):
        print(f"Closing mwc {self.name}")
        self.outfile.close()
        print(f"Closed mwc {self.name}")


def get_next_index():
    dirname = os.path.dirname(args.output)
    dirname = "." if dirname == "" else dirname
    basename = os.path.basename(args.output)

    regex = re.compile(re.escape(basename) + "_(\d+)\..*")
    matches = (regex.match(fn) for fn in os.listdir(dirname))
    indexes = tuple((int(m.group(1)) for m in matches if m))
    max_index = -1 if len(indexes) == 0 else max(indexes)

    yield from count(max_index + 1)


def get_fd():
    for idx in get_next_index():
        fn = f"{args.output}_{idx:015d}"
        if args.lz4:
            outfile = lz4f.open(f"{fn}.lz4", mode='wt', encoding='utf-8', compression_level=16)
        elif args.gzip:
            outfile = gzip.open(f"{fn}.gz", "wt", encoding='utf-8')
        else:
            outfile = open(args.output, "w")
        yield (idx, fn, outfile)


def run():
    try:
        old_wsclient = None
        last_count, last_tick, last_block_count = 0, 0, 0
        starting_at = datetime.datetime.now()
        for block_idx, fn, outfile in get_fd():
            # block_idx, fn, outfile = get_fd()
            print(f"Starting block {block_idx}: {fn}")
            wsClient = MyWebsocketClient(fn, outfile, count=last_count)
            wsClient.start()
            print(wsClient.url, wsClient.products)
            print("\nCollecting Data")
            same_block = True
            while same_block:
                message_count, now = wsClient.message_count, datetime.datetime.now()
                elapsed = (now - starting_at).seconds
                global_rate = message_count / elapsed if elapsed > 0 else 0
                point_count = (message_count - last_count)
                point_rate = point_count / (now - last_tick).seconds if last_tick != 0 else 0

                if point_rate == 0 and (message_count - last_block_count) > 0 and last_tick != 0:
                    raise NoDataException  # we are not receiving data. Kill ourselves
                elif old_wsclient is not None and (message_count - last_block_count) > 10000:
                    print(f"Closing old block {old_wsclient.name}...")
                    old_wsclient.close()
                    print(f"Closed old block {old_wsclient.name}!")
                    old_wsclient = None

                last_count, last_tick = message_count, now

                print(f"Now {point_count:13d} ({point_rate:7.1f})   | Global {message_count:13d} ({global_rate:7.1f})")
                if (message_count - last_block_count) > args.blocksize:
                    print(f"Reached block limit {args.blocksize}")
                    last_block_count = message_count
                    same_block = False
                time.sleep(5)
            print(f"New block!!!")
            old_wsclient = wsClient
    except (KeyboardInterrupt, NoData, WebSocketTimeoutException) as e:
        print(e)
        if old_wsclient is not None:
            old_wsclient.close()

    print(f"Collected {wsClient.message_count} msgs")
    time.sleep(1)
    print("Closing connection... ", end="")
    try:
        wsClient.close()
        print("OK!")
    except Exception as e:
        print(" Already closed?", e)
        wsClient.ws.close()
        wsClient.on_close()
        outfile.close()
        wsClient.stop = True
        wsClient.thread.join()
    print("Waiting for joining thread...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GDAX data saver')
    parser.add_argument('output', metavar='outputfile', type=str,
                        help='name of the output file')

    parser.add_argument('-z', dest='gzip', action='store_true', help='use gzip to compress', default=True)
    parser.add_argument('-l', dest='lz4', action='store_true', help='use lz4 to compress', default=False)
    parser.add_argument('--block-size', dest='blocksize', default=1000000, help='block size', type=int)

    args = parser.parse_args()

    run()
    sys.exit()
