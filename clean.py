#! /usr/bin/python3

import ciso8601
from datetime import datetime
import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

# This code cleans the enron and seattle datasets.
#    - Removes non-metadata fields.
#    - Discards messages where sender/recipient data is unknown.
#    - Discards messages where there is more than one sender/recipient. We are interested in point to point messages only.
#    - Parses times into milliseconds since the Unix epoch.
#    - Discards messages where the timestamp is outside the known range for each dataset.
#    - Preserves the ordering of messages per user by incrementing messages with equal timestamps by 1 millisecond.
#    - Reorganizes messages so that they can be looked up per user.
#    - Strips extraneous characters from senders and receivers for a basic approach to entity resolution.

def clean_enron_sender(s):
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

def clean_enron_receiver(s):
    s = clean_enron_sender(s)
    s = s.split(",")

    if len(s) != 1:
        return ""
    else:
        return s[0]

def clean_seattle_sender(s):
    if type(s) is str:
        s = s.split("<")[0]
        s = s.split("(")[0]
        s = s.replace("\"", "")
        s = s.replace("\'", "")
        s = s.replace(" ", "")

        return s
    else:
        return ""

def clean_seattle_receiver(s):
    if type(s) is str:
        s = s.split(";")
        if len(s) != 1:
            return ""
        else:
            return clean_seattle_sender(s[0])
    else:
        return ""

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

def correct(df, start, end):
    sender_correct = df.sender.apply(lambda x: x != "" and x.lower() != "nan")
    receiver_correct = df.receiver.apply(lambda x: x != "" and x.lower() != "nan")
    cc_correct = df.cc.apply(lambda x: type(x) is not str or x.lower() == "nan")
    bcc_correct = df.bcc.apply(lambda x: type(x) is not str or x.lower() == "nan")
    submit_correct = df.submit.apply(lambda x: start <= x and x <= end)

    all_correct = sender_correct & receiver_correct & cc_correct & bcc_correct & submit_correct
    return df[all_correct]

def clean(df, start, end, sender_clean, receiver_clean):
    df.sender = df.sender.apply(sender_clean)
    df.receiver = df.receiver.apply(receiver_clean)
    df.submit = df.submit.apply(convert_time)
    df = correct(df, start, end)

    return df[["sender", "receiver", "submit"]]

def make_unique_ascending(m, col):
    m = m[m[:, col].argsort()]
    prev = m[0, col]
    for idx in range(1, m.shape[0]):
        if m[idx, col] <= prev:
            prev += 1
            m[idx, col] = prev
        else:
            prev = m[idx, col]
    return m

def users(df):
    users = np.unique(df[:, 0])
    submits = np.empty(len(users), dtype=object)
    receivers = np.empty(len(users), dtype=object)
    for i in tqdm(range(len(users))):
        user = users[i]
        idx = df[:, 0] == user
        m = make_unique_ascending(df[idx, 1:], 1)
        receivers[i] = m[:, 0]
        submits[i] = m[:, 1]
    return pd.DataFrame({"sender": users, "receivers": receivers, "submits": submits})

def main(data_path):
    enron_raw_path = os.path.join(data_path, "enron_raw.csv")
    enron_clean_path = os.path.join(data_path, "enron_clean.csv")
    enron_path = os.path.join(data_path, "enron_processed.csv")

    seattle_raw_path = os.path.join(data_path, "seattle_raw.csv")
    seattle_clean_path = os.path.join(data_path, "seattle_clean.csv")
    seattle_path = os.path.join(data_path, "seattle_processed.csv")

    if not os.path.exists(enron_path):
        print("Creating: %s..." % (enron_path))
        if not os.path.exists(enron_clean_path):
            print("Creating: %s..." % (enron_clean_path))
            enron_raw_df = pd.read_csv(enron_raw_path, usecols=["From", "To", "X-cc", "X-bcc", "Date"])
            enron_raw_df.rename(columns={"From": "sender", "To": "receiver", "X-cc": "cc", "X-bcc": "bcc", "Date": "submit"}, inplace=True)

            enron_start = 490320000 # January 16, 1985
            enron_end = 1007337600 # December 3, 2001
            clean_enron_df = clean(enron_raw_df, enron_start, enron_end, clean_enron_sender, clean_enron_receiver)
            print("done.")
            clean_enron_df.to_csv(enron_clean_path, index=False)
        else:
            clean_enron_df = pd.read_csv(enron_clean_path)

        clean_enron_m = clean_enron_df.to_numpy()
        enron_users = users(clean_enron_m)
        print("done.")
        enron_users.to_json(enron_path)

    if not os.path.exists(seattle_path):
        print("Creating: %s..." % (seattle_path))
        if not os.path.exists(seattle_clean_path):
            print("Creating: %s..." % (seattle_clean_path))
            seattle_raw_df = pd.read_csv(seattle_raw_path, usecols=["sender", "to", "cc", "bcc", "time"])
            seattle_raw_df.rename(columns={"to": "receiver", "time": "submit"}, inplace=True)

            seattle_start = 1483228800 # January 1, 2017
            seattle_end = 1491004800 # April 1, 2017
            clean_seattle_df = clean(seattle_raw_df, seattle_start, seattle_end, clean_seattle_sender, clean_seattle_receiver)
            print("done.")
            clean_seattle_df.to_csv(seattle_clean_path, index=False)
        else:
            clean_seattle_df = pd.read_csv(seattle_clean_path, dtype={"sender": str, "submit": int})

        clean_seattle_df = clean_seattle_df.to_numpy()
        seattle_users = users(clean_seattle_df)
        print("done.")
        seattle_users.to_json(seattle_path)

if __name__ == "__main__":
    data_path = os.path.abspath(sys.argv[1])
    main(data_path)