#! /usr/bin/python3

import argparse
import ciso8601
import numpy as np
import os
import pandas as pd
import requests
from tqdm import tqdm
import zipfile

# This code cleans the enron and seattle datasets.
#    - Removes non-metadata fields.
#    - Discards messages where sender/recipient data is unknown.
#    - Discards messages where there is more than one sender/recipient. We are interested in point to point messages only.
#    - Parses times into milliseconds since the Unix epoch.
#    - Discards messages where the timestamp is outside the known range for each dataset.
#    - Preserves the ordering of messages per user by incrementing messages with equal timestamps by 1 millisecond.
#    - Reorganizes messages so that they can be looked up per user.
#    - Strips extraneous characters from senders and receivers for a basic approach to entity resolution.

def clean_enron_sender_o2o(s):
    if type(s) is str:
        s = s.replace("[", "")
        s = s.replace("]", "")
        s = s.replace("\'", "")
        s = s.replace("\"", "")
        s = s.replace(" ", "")
        s = s.replace(".", "")

        return s
    else:
        return ""

def clean_enron_receiver_o2o(s):
    s = clean_enron_sender_o2o(s)
    s = s.split(",")

    if len(s) != 1:
        return ""
    else:
        return s[0]

def clean_seattle_sender_o2o(s):
    if type(s) is str:
        s = s.split("<")[0]
        s = s.split("(")[0]
        s = s.replace("\"", "")
        s = s.replace("\'", "")
        s = s.replace(" ", "")

        return s
    else:
        return ""

def clean_seattle_receiver_o2o(s):
    if type(s) is str:
        s = s.split(";")
        if len(s) != 1:
            return ""
        else:
            return clean_seattle_sender_o2o(s[0])
    else:
        return ""

def clean_enron_sender(s):
    pass

def clean_enron_receiver(s):
    pass

def clean_seattle_sender(s):
    pass

def clean_seattle_receiver(s):
    pass


def convert_time(s):
    """
    Converts string times to a standard float datetime.
    """
    try:
        (date, time) = s.split(" ")
        iso = date + "T" + time
        dt = ciso8601.parse_datetime(iso)
        return int(np.round(dt.timestamp()))
    except:
        return -1

def correct_o2o(df, start, end):
    sender_correct = df.sender.apply(lambda x: x != "" and x.lower() != "nan")
    receiver_correct = df.receiver.apply(lambda x: x != "" and x.lower() != "nan")
    cc_correct = df.cc.apply(lambda x: type(x) is not str or x.lower() == "nan")
    bcc_correct = df.bcc.apply(lambda x: type(x) is not str or x.lower() == "nan")

    all_correct = sender_correct & receiver_correct & cc_correct & bcc_correct
    return df[all_correct]

def clean_o2o(df, start, end, sender_clean, receiver_clean):
    df.submit = df.submit.apply(convert_time)
    df = df[df.submit.apply(lambda x: start <= x and x <= end)]

    df.sender = df.sender.apply(sender_clean)
    df.receiver = df.receiver.apply(receiver_clean)
    df = correct_o2o(df, start, end)

    return df[["sender", "receiver", "submit"]]

def clean(df, start, end, sender_clean, receiver_clean):
    pass

def sort_matrix(m, col):
    m = m[m[:, col].argsort()]
    return m

def users(df):
    users = np.unique(df[:, 0])
    submits = np.empty(len(users), dtype=object)
    receivers = np.empty(len(users), dtype=object)
    for i in tqdm(range(len(users))):
        user = users[i]
        idx = df[:, 0] == user
        m = sort_matrix(df[idx, 1:], 1)
        receivers[i] = m[:, 0]
        submits[i] = m[:, 1]
        df = np.delete(df, idx, axis=0)

    return pd.DataFrame({"sender": users, "receivers": receivers, "submits": submits})


def download(url, file_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as file:
            for d in r.iter_content(chunk_size=8192):
                file.write(d)

def decompress(in_path, out_path):
    with zipfile.ZipFile(in_path, "r") as zip:
        zip.extractall(out_path)

def process(url, zip_path, clean_path, processed_path, rename, start, end, clean_fn, clean_sender_fn, clean_receiver_fn):
    if not os.path.exists(processed_path):
        print(f"Creating: {processed_path}...")

        if not os.path.exists(clean_path):
            print(f"Creating: {clean_path}...")

            if not os.path.exists(zip_path):
                print(f"Downloading: {url}...")
                download(url, zip_path)
                print("done.")

            raw_df = pd.read_csv(zip_path, compression="zip")
            raw_df.rename(columns=rename, inplace=True)
            clean_df = clean_fn(raw_df, start, end, clean_sender_fn, clean_receiver_fn)
            print("done.")

            clean_df.to_csv(clean_path, index=False)
        else:
            clean_df = pd.read_csv(clean_path, dtype={"sender": str, "submit": int})

        clean_m = clean_df.to_numpy()
        processed = users(clean_m)
        print("done.")
        processed.to_json(processed_path)

def main(data_path, s):

    enron_raw_url = "https://files.ssrc.us/data/enron.zip"
    enron_zip_path = os.path.join(data_path, "enron.zip")
    enron_clean_path = os.path.join(data_path, f"enron_clean_{'s' if s else ''}.csv")
    enron_processed_path = os.path.join(data_path, f"enron_processed_{'s' if s else ''}.csv")
    enron_rename = {"From": "sender", "X-From": "xsender", "X-To": "xreceiver", "To": "receiver", "X-cc": "cc", "X-bcc": "bcc", "Date": "submit"}
    enron_start = 490320000 # January 16, 1985
    enron_end = 1007337600 # December 3, 2001

    seattle_raw_url = "https://files.ssrc.us/data/seattle.zip"
    seattle_zip_path = os.path.join(data_path, "seattle.zip")
    seattle_clean_path = os.path.join(data_path, f"seattle_clean_{'s' if s else ''}.csv")
    seattle_processed_path = os.path.join(data_path, f"seattle_processed_{'s' if s else ''}.csv")
    seattle_rename = {"to": "receiver", "time": "submit"}
    seattle_start = 1483228800 # January 1, 2017
    seattle_end = 1491004800 # April 1, 2017

    if s: # One-to-one communications.
        enron_clean_fn = clean_o2o
        enron_clean_sender_fn = clean_enron_sender_o2o
        enron_clean_receiver_fn = clean_enron_receiver_o2o

        seattle_clean_fn = clean_o2o
        seattle_clean_sender_fn = clean_seattle_sender_o2o
        seattle_clean_receiver_fn = clean_seattle_receiver_o2o
    else: # One-to-many communications.
        enron_clean_fn = clean
        enron_clean_sender_fn = clean_enron_sender
        enron_clean_receiver_fn = clean_enron_receiver

        seattle_clean_fn = clean
        seattle_clean_sender_fn = clean_seattle_sender
        seattle_clean_receiver_fn = clean_seattle_receiver

    process(enron_raw_url, enron_zip_path, enron_clean_path, enron_processed_path, enron_rename, enron_start, enron_end,
            enron_clean_fn, enron_clean_sender_fn, enron_clean_receiver_fn)

    # process(seattle_raw_url, seattle_zip_path, seattle_clean_path, seattle_processed_path, seattle_rename, seattle_start, seattle_end,
    #         seattle_clean_fn, seattle_clean_sender_fn, seattle_clean_receiver_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Metadata Cleaner",
                                     description="This code downloads and cleans the Enron and Seattle Email datasets.")
    parser.add_argument("path", type=str, help="Data path to look for data files and store generated data files.")
    parser.add_argument("-s", action="store_true", help="If flag is specified, the cleaning process will ensure each recipient has exactly one receiver.")
    args = parser.parse_args()

    data_path = os.path.abspath(args.path)
    s = args.s

    main(data_path, s)