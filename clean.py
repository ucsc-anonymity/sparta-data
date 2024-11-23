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


ENRON_PARAMETERS = (
    "https://files.ssrc.us/data/enron.zip",  # url
    "enron",  # path
    ",",  # delimiter
    {  # rename
        "From": "sender",
        "To": "receiver",
        "X-cc": "cc",
        "X-bcc": "bcc",
        "Date": "submit"
    },
    clean_enron,  # cleaning function
    490338000,  # start: July 16, 1985
    1007337600,  # end: December 3, 2001
)

SEATTLE_PARAMETERS = (
    "https://files.ssrc.us/data/seattle.zip",  # url
    "seattle",  # path
    ";",  # delimiter
    {  # rename
        "sender": "sender",
        "to": "receiver",
        "cc": "cc",
        "bcc": "bcc",
        "time": "submit"
    },
    clean_seattle,  # cleaning function
    1483228800,  # start: January 1, 2017
    1491004800,  # end: April 1, 2017
)


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


def clean_multiple(s, clean_fn, delimiter):
    """
    Cleans multiple user names.

    s - the user names in a string.
    clean_fn - the function to clean individual user names.
    delimiter - the string that marks the end of a user name.
    """
    if type(s) is str:
        s = [clean_fn(r) for r in s.split(delimiter)]

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


def clean(df, start, end, clean_fn, delimiter):
    """
    Cleans the dataset. Ensures that sender and receiver information is set, and
    times are between start and end. Then factorizes the senders and receivers.
    Returns a cleaned dataframe.
    """
    print("initial: ", df.shape)
    df.sender = df.sender.apply(clean_fn)
    df.receiver = df.receiver.apply(lambda x:
                                    clean_multiple(x, clean_fn, delimiter))
    df.submit = df.submit.apply(convert_time)

    reorder_cols = ["sender", "receiver", "submit", "cc", "bcc"]
    df = df[reorder_cols].to_numpy()
    df = np.concatenate([expand(df[i, :]) for i in range(df.shape[0])], axis=0)
    print("expansion: ", df.shape)

    df = pd.DataFrame(df, columns=reorder_cols)
    sender_correct = df.sender.apply(lambda x: x != "" and x.lower() != "nan")
    print("senders: ", sum(sender_correct), len(sender_correct))
    receiver_correct = df.receiver.apply(
        lambda x: x != "" and x.lower() != "nan")
    print("receivers: ", sum(receiver_correct), len(receiver_correct))
    time_correct = df.submit.apply(lambda x: start <= x and x <= end)
    print("time: ", sum(time_correct), len(time_correct))
    all_correct = sender_correct & receiver_correct & time_correct
    df = df[all_correct]
    print("final: ", df.shape)

    stacked = df[["sender", "receiver"]].stack()
    sender_receiver, user_key = stacked.factorize()
    print("users: ", np.unique(user_key).shape)
    df[["sender", "receiver"]] = pd.Series(
        sender_receiver, index=stacked.index).unstack()
    clean_cols = ["sender", "receiver", "submit"]

    user_key = pd.DataFrame(user_key, columns=["user"])
    df = df[clean_cols]
    df.sort_values(by=["submit", "sender", "receiver"], inplace=True)

    return df, user_key


# data_path, *parameters
def process(data_path, url, dataset_name, delimiter, rename, clean_sender_fn, start, end):
    """
    Processes the datasets into a more usable form.
    """

    dataset_path = os.path.join(data_path, dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    zip_path = os.path.join(dataset_path, "raw.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading: {url}...")
        download(url, zip_path, dataset_path)
        print("done.")

    clean_path = os.path.join(
        dataset_path, f"clean.csv")
    if not os.path.exists(clean_path):
        print(f"Creating: {clean_path}...")

        raw_path = os.path.join(dataset_path, "raw.csv")
        raw_df = pd.read_csv(raw_path, usecols=rename.keys())
        raw_df.rename(columns=rename, inplace=True)
        clean_df, user_key = clean(
            raw_df, start, end, clean_sender_fn, delimiter)
        print("done.")

        clean_df.to_csv(clean_path, index=False)

        user_key_path = os.path.join(
            dataset_path, f"users.csv")
        user_key.to_csv(user_key_path)
    else:
        clean_df = pd.read_csv(clean_path)


def main(data_path, enron, seattle):
    """
    Downloads the datasets if not available, then cleans and processes them.
    """

    if enron:
        parameters = ENRON_PARAMETERS
    elif seattle:
        parameters = SEATTLE_PARAMETERS
    else:
        raise ValueError("Unrecognized dataset. Expects `enron` or `seattle`.")

    process(data_path, *parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Metadata Cleaner", description="This code downloads and cleans the Enron and Seattle Email datasets.")
    parser.add_argument(
        "path", type=str, help="Data path to look for data files and store generated data files.")
    parser.add_argument("--enron", action="store_true",
                        help="Generate enron data.")
    parser.add_argument("--seattle", action="store_true",
                        help="Generate seattle data.")
    args = parser.parse_args()

    data_path = os.path.abspath(args.path)

    main(data_path, args.enron, args.seattle)
