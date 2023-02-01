#! /usr/bin/python3

import argparse
import ciso8601
import numpy as np
import os
import pandas as pd
import requests
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

def download(url, file_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as file:
            for d in r.iter_content(chunk_size=8192):
                file.write(d)

def clean_enron(s):
    if type(s) is str:
        s = s.replace("[", "")
        s = s.replace("]", "")
        s = s.replace("\'", "")
        s = s.replace("\"", "")
        s = s.replace(" ", "")

        return s
    else:
        return ""

def clean_seattle(s):
    if type(s) is str:
        s = s.split("<")[0]
        s = s.split("(")[0]
        s = s.replace("\"", "")
        s = s.replace("\'", "")
        s = s.replace(" ", "")

        return s
    else:
        return ""

def clean_multiple(s, clean_fn, delimiter, single_senders):
    if type(s) is str:
        s = [clean_fn(r) for r in s.split(delimiter)]

        if single_senders and len(s) != 1:
            return []
        else:
            return s
    else:
        return []

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

def expand(row):
    r = np.empty((len(row[1]), len(row)), dtype=object)
    r[:] = row
    for i in range(r.shape[0]):
        r[i, 1] = row[1][i]
    return r

def correct(df, single_senders):
    sender_correct = df.sender.apply(lambda x: x != "" and x.lower() != "nan")
    receiver_correct = df.receiver.apply(lambda x: x != "" and x.lower() != "nan")

    all_correct = sender_correct & receiver_correct

    if single_senders:
        cc_correct = df.cc.apply(lambda x: type(x) is not str or x.lower() == "nan")
        bcc_correct = df.bcc.apply(lambda x: type(x) is not str or x.lower() == "nan")
        all_correct &= cc_correct & bcc_correct

    return df[all_correct]

def clean(df, start, end, clean_fn, delimiter, single_senders):
    df.sender = df.sender.apply(clean_fn)
    df.receiver = df.receiver.apply(lambda x: clean_multiple(x, clean_fn, delimiter, single_senders))

    reorder_cols = ["sender", "receiver", "submit", "cc", "bcc"]
    df = df[reorder_cols].to_numpy()
    df = np.concatenate([expand(df[i, :]) for i in range(df.shape[0])], axis=0)
    df = pd.DataFrame(df, columns=reorder_cols)
    df = correct(df, single_senders)
    df.submit = df.submit.apply(convert_time)
    df = df[df.submit.apply(lambda x: start <= x and x <= end)]

    df.sender, senders = df.sender.factorize()
    df.receiver, receivers = df.receiver.factorize()

    senders = pd.DataFrame(senders)
    receivers = pd.DataFrame(receivers)

    clean_cols = ["sender", "receiver", "submit"]
    return df[clean_cols], senders, receivers

def users(df):
    df = df[np.lexsort((df[:, 2], df[:, 0]))]

    users = np.unique(df[:, 0])
    receivers = np.empty(len(users), dtype=object)
    submits = np.empty(len(users), dtype=object)

    min = 0
    for i in range(df.shape[0]):
        if df[i, 0] != df[min, 0]:
            receivers[df[min, 0]] = df[min:i, 1]
            submits[df[min, 0]] = df[min:i, 2]
            min = i
    receivers[df[min, 0]] = df[min:df.shape[0], 1]
    submits[df[min, 0]] = df[min:df.shape[0], 2]

    return pd.DataFrame({"sender": users, "receivers": receivers, "submits": submits})

def process(url, zip_path, clean_path, senders_path, receivers_path,
            processed_path, rename, start, end, clean_sender_fn, delimiter, single_senders):
    if not os.path.exists(processed_path):
        print(f"Creating: {processed_path}...")

        if not os.path.exists(clean_path):
            print(f"Creating: {clean_path}...")

            if not os.path.exists(zip_path):
                print(f"Downloading: {url}...")
                download(url, zip_path)
                print("done.")

            raw_df = pd.read_csv(zip_path, usecols=rename.keys() ,compression="zip")
            raw_df.rename(columns=rename, inplace=True)
            clean_df, senders, receivers = clean(raw_df, start, end, clean_sender_fn, delimiter, single_senders)
            print("done.")

            clean_df.to_csv(clean_path, index=False)
            senders.to_csv(senders_path)
            receivers.to_csv(receivers_path)
        else:
            clean_df = pd.read_csv(clean_path)

        clean_m = clean_df.to_numpy()
        processed = users(clean_m)
        print("done.")
        processed.to_json(processed_path)

def main(data_path, single_senders):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    enron_data_path = os.path.join(data_path, "enron")
    if not os.path.exists(enron_data_path):
        os.makedirs(enron_data_path)

    seattle_data_path = os.path.join(data_path, "seattle")
    if not os.path.exists(seattle_data_path):
        os.makedirs(seattle_data_path)

    enron_raw_url = "https://files.ssrc.us/data/enron.zip"
    enron_zip_path = os.path.join(enron_data_path, "enron.zip")
    enron_clean_path = os.path.join(enron_data_path, f"clean{'_s' if single_senders else ''}.csv")
    enron_senders_path = os.path.join(enron_data_path, f"senders{'_s' if single_senders else ''}.csv")
    enron_receivers_path = os.path.join(enron_data_path, f"receivers{'_s' if single_senders else ''}.csv")
    enron_processed_path = os.path.join(enron_data_path, f"processed{'_s' if single_senders else ''}.csv")
    enron_rename = {"From": "sender", "To": "receiver", "X-cc": "cc", "X-bcc": "bcc", "Date": "submit"}
    enron_start = 490320000 # January 16, 1985
    enron_end = 1007337600 # December 3, 2001

    seattle_raw_url = "https://files.ssrc.us/data/seattle.zip"
    seattle_zip_path = os.path.join(seattle_data_path, "seattle.zip")
    seattle_clean_path = os.path.join(seattle_data_path, f"clean{'_s' if single_senders else ''}.csv")
    seattle_senders_path = os.path.join(seattle_data_path, f"senders{'_s' if single_senders else ''}.csv")
    seattle_receivers_path = os.path.join(seattle_data_path, f"receivers{'_s' if single_senders else ''}.csv")
    seattle_processed_path = os.path.join(seattle_data_path, f"processed{'_s' if single_senders else ''}.csv")
    seattle_rename = {"sender": "sender", "to": "receiver", "cc": "cc", "bcc": "bcc", "time": "submit"}
    seattle_start = 1483228800 # January 1, 2017
    seattle_end = 1491004800 # April 1, 2017

    process(enron_raw_url, enron_zip_path, enron_clean_path, enron_senders_path, enron_receivers_path,
            enron_processed_path, enron_rename, enron_start, enron_end, clean_enron, ",", single_senders)

    process(seattle_raw_url, seattle_zip_path, seattle_clean_path, seattle_senders_path, seattle_receivers_path,
            seattle_processed_path, seattle_rename, seattle_start, seattle_end, clean_seattle, ";", single_senders)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Metadata Cleaner",
                                     description="This code downloads and cleans the Enron and Seattle Email datasets.")
    parser.add_argument("path", type=str, help="Data path to look for data files and store generated data files.")
    parser.add_argument("-s", action="store_true", help="If flag is specified, the cleaning process will ensure each recipient has exactly one receiver.")
    args = parser.parse_args()

    data_path = os.path.abspath(args.path)
    single_senders = args.s

    main(data_path, single_senders)