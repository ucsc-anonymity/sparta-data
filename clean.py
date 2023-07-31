#! /usr/bin/python3

import argparse
import ciso8601
import numpy as np
import os
import pandas as pd
import requests
from zipfile import ZipFile

# This code cleans the enron and seattle datasets.
#    - Removes non-metadata fields.
#    - Discards messages where sender/recipient data is unknown.
#    - Discards messages where there is more than one sender/recipient. We are
#    interested in point to point messages only.
#    - Parses times into milliseconds since the Unix epoch.
#    - Discards messages where the timestamp is outside the known range for each
#    dataset.
#    - Preserves the ordering of messages per user by incrementing messages with
#    equal timestamps by 1 millisecond.
#    - Reorganizes messages so that they can be looked up per user.
#    - Strips extraneous characters from senders and receivers for a basic
#    approach to entity resolution.


def download(url, file_path, data_path):
    """
    Downloads the file at url and saves it to file_path.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as file:
            for d in r.iter_content(chunk_size=8192):
                file.write(d)

    with ZipFile(file_path) as z:
        z.extractall(path=data_path)


def clean_enron(s):
    """
    Cleans Enron user names.
    """
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
    """
    Cleans Seattle user names.
    """
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
    """
    Cleans multiple user names.

    s - the user names in a string.
    clean_fn - the function to clean individual user names.
    delimiter - the string that marks the end of a user name.
    single_sender - a flag to indicate whether multiple user names are allowed,
        in which case they are parsed into an array.
    """
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
    """
    Expands a tuple with multiple receivers into a data frame,
    where the sender and time are constant and each row has a single receiver.
    """
    r = np.empty((len(row[1]), len(row)), dtype=object)
    r[:] = row
    for i in range(r.shape[0]):
        r[i, 1] = row[1][i]
    return r


def correct(df, single_senders):
    """
    Filters datasets so that sender and receiver information is correct, and if
    single sender is set, then the cc and bcc fields are invalid.
    """
    sender_correct = df.sender.apply(lambda x: x != "" and x.lower() != "nan")
    receiver_correct = df.receiver.apply(lambda x: x != "" and x.lower() != "nan")

    all_correct = sender_correct & receiver_correct

    if single_senders:
        cc_correct = df.cc.apply(lambda x: type(x) is not str or x.lower() == "nan")
        bcc_correct = df.bcc.apply(lambda x: type(x) is not str or x.lower() == "nan")
        all_correct &= cc_correct & bcc_correct

    return df[all_correct]


def clean(df, start, end, clean_fn, delimiter, single_senders):
    """
    Cleans the dataset. Ensures that sender and receiver information is set,
    times are between start and end, and that single senders is enforced if it
    is set. Then factorizes the senders and receivers. Returns a cleaned
    """
    df.sender = df.sender.apply(clean_fn)
    df.receiver = df.receiver.apply(lambda x: clean_multiple(
        x, clean_fn, delimiter, single_senders))

    reorder_cols = ["sender", "receiver", "submit", "cc", "bcc"]
    df = df[reorder_cols].to_numpy()
    df = np.concatenate([expand(df[i, :]) for i in range(df.shape[0])], axis=0)
    df = pd.DataFrame(df, columns=reorder_cols)
    df = correct(df, single_senders)
    df.submit = df.submit.apply(convert_time)
    df = df[df.submit.apply(lambda x: start <= x and x <= end)]

    stacked = df[["sender", "receiver"]].stack()
    sender_receiver, user_key = stacked.factorize()
    df[["sender", "receiver"]] = pd.Series(sender_receiver, index=stacked.index).unstack()
    clean_cols = ["sender", "receiver", "submit"]

    user_key = pd.DataFrame(user_key)
    df = df[clean_cols]
    df.sort_values(by=["submit"], inplace=True)

    return df, user_key


def senders(df):
    """
    Process the clean datasets so that for each sender we have a sorted list of
    when they sent messages and who they sent to.
    """
    # df[0] - sender
    # df[1] - receiver
    # df[2] - time
    # Sorts by sender then submit time.
    df = df[np.lexsort((df[:, 2], df[:, 0]))]

    users = np.unique(df[:, 0])
    receivers = np.empty(len(users), dtype=object)
    submits = np.empty(len(users), dtype=object)

    start = 0
    end = 0
    for i in range(len(users)):
        while end < df.shape[0] and df[start, 0] == df[end, 0]:
            end += 1
        receivers[i] = df[start:end, 1]
        submits[i] = df[start:end, 2]
        start = end

    return pd.DataFrame({"sender": users, "receivers": receivers, "submits": submits})


def receivers(df):
    """
    Processes the clean datasets so that for each receiver we have a sorted list
    of when they were sent messages and who sent them.
    """
    # df[0] - sender
    # df[1] - receiver
    # df[2] - time
    # Sorts by receiver then submit time.
    df = df[np.lexsort((df[:, 2], df[:, 1]))]

    users = np.unique(df[:, 1])
    senders = np.empty(len(users), dtype=object)
    submits = np.empty(len(users), dtype=object)

    start = 0
    end = 0
    for i in range(len(users)):
        while end < df.shape[0] and df[start, 1] == df[end, 1]:
            end += 1
        senders[i] = df[start:end, 0]
        submits[i] = df[start:end, 2]
        start = end

    return pd.DataFrame({"receiver": users, "senders": senders, "submits": submits})


def process(url, data_path, rename, start, end, clean_sender_fn, delimiter, single_senders):
    """
    Processes the datasets into a more usable form.
    """
    raw_path = os.path.join(data_path, "raw.csv")
    zip_path = os.path.join(data_path, "raw.zip")
    clean_path = os.path.join(data_path, f"clean{'_s' if single_senders else ''}.csv")
    user_key_path = os.path.join(data_path, f"users{'_s' if single_senders else ''}.csv")
    senders_processed_path = os.path.join(
        data_path, f"senders_processed{'_s' if single_senders else ''}.csv")
    receivers_processed_path = os.path.join(
        data_path, f"receivers_processed{'_s' if single_senders else ''}.csv")

    if not os.path.exists(zip_path):
        print(f"Downloading: {url}...")
        download(url, zip_path, data_path)
        print("done.")

    if not os.path.exists(clean_path):
        print(f"Creating: {clean_path}...")

        raw_df = pd.read_csv(raw_path, usecols=rename.keys())
        raw_df.rename(columns=rename, inplace=True)
        clean_df, user_key = clean(raw_df, start, end, clean_sender_fn, delimiter, single_senders)
        print("done.")

        clean_df.to_csv(clean_path, index=False)
        user_key.to_csv(user_key_path)
    else:
        clean_df = pd.read_csv(clean_path)
    clean_m = clean_df.to_numpy()

    if not os.path.exists(senders_processed_path):
        print(f"Creating: {senders_processed_path}...")
        senders_processed = senders(clean_m)
        print("done.")
        senders_processed.to_json(senders_processed_path)

    if not os.path.exists(receivers_processed_path):
        print(f"Creating: {receivers_processed_path}...")
        receivers_processed = receivers(clean_m)
        print("done.")
        receivers_processed.to_json(receivers_processed_path)


def active_users(df, min_percentile, max_percentile):
    """
    Filters a dataframe, df, such that each user is in the min_percentile to
    max_percentile range based on number of messages submitted.
    """

    lens = df.submits.apply(len).to_numpy()
    min = np.percentile(lens, min_percentile)
    max = np.percentile(lens, max_percentile)

    return df[(lens >= min) & (lens <= max)]


def main(data_path, single_senders=False, enron=False, seattle=False):
    """
    Downloads the datasets if not available, then cleans and processes them.
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    enron_data_path = os.path.join(data_path, "enron")
    if not os.path.exists(enron_data_path):
        os.makedirs(enron_data_path)

    seattle_data_path = os.path.join(data_path, "seattle")
    if not os.path.exists(seattle_data_path):
        os.makedirs(seattle_data_path)

    enron_raw_url = "https://files.ssrc.us/data/enron.zip"
    enron_rename = {"From": "sender", "To": "receiver",
                    "X-cc": "cc", "X-bcc": "bcc", "Date": "submit"}
    enron_start = 490320000  # January 16, 1985
    enron_end = 1007337600  # December 3, 2001

    seattle_raw_url = "https://files.ssrc.us/data/seattle.zip"
    seattle_rename = {"sender": "sender", "to": "receiver",
                      "cc": "cc", "bcc": "bcc", "time": "submit"}
    seattle_start = 1483228800  # January 1, 2017
    seattle_end = 1491004800  # April 1, 2017

    if enron:
        process(enron_raw_url, enron_data_path, enron_rename,
                enron_start, enron_end, clean_enron, ",", single_senders)

    if seattle:
        process(seattle_raw_url, seattle_data_path, seattle_rename,
                seattle_start, seattle_end, clean_seattle, ";", single_senders)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Metadata Cleaner", description="This code downloads and cleans the Enron and Seattle Email datasets.")
    parser.add_argument(
        "path", type=str, help="Data path to look for data files and store generated data files.")
    parser.add_argument("--single-receiver", action="store_true",
                        help="The cleaning process will ensure each message has exactly one receiver.")
    parser.add_argument("--enron", action="store_true", help="Generate enron data.")
    parser.add_argument("--seattle", action="store_true", help="Generate seattle data.")
    args = parser.parse_args()

    data_path = os.path.abspath(args.path)

    main(data_path, args.single_receiver, args.enron, args.seattle)
