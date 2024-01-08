# -*- coding: utf-8 -*-
import copy

import numpy as np
import pandas as pd

import utils_op as utl_op


def generate_sublevel_index(dft: pd.DataFrame, value_col: str = "value", annotation_col: str = "correcteness"):
    columns_multiindex = pd.MultiIndex.from_product(
        [dft.columns, [value_col, annotation_col]], names=['column', None])

    # Create a new DataFrame with the MultiIndex columns
    df_new = pd.DataFrame(columns=columns_multiindex)

    # Fill in the 'value' subcolumns with the original values
    df_new[[(col, value_col) for col in dft.columns]] = dft.values
    return df_new


def generate_excel_file_to_annotate_from_generated_data_with_azure(
        input_path: str = "./data/b_intermediate/fictifs_cr.pickle",
        output_path='./data/b_intermediate/synthetic_data_azure_to_annotate.xlsx',
        add_time_stamp_to_output: bool = False):

    df = pd.read_pickle(input_path)
    i = 0
    l_df_rows = []
    for i in range(df.shape[0]):
        res_dc = {}
        res_dc["CR Fictif"] = df.loc[:, "fictive_cr"].iloc[i]
        try:
            res_dc.update(eval(df.loc[:, "summaries"].iloc[i]))
        except:
            print(f"ERROR with summaries parsing in element {i}")
        try:
            res_dc.update(eval(df.loc[:, "ner_from_cr"].iloc[i]))
        except:
            print(f"ERROR with ner_from_cr parsing in element {i}")

        l_df_rows.append(copy.deepcopy(res_dc))

    dft = pd.DataFrame(l_df_rows)
    df_new = generate_sublevel_index(dft)

    if add_time_stamp_to_output:
        output_path = utl_op.add_timestamp_to_filename(output_path)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df_new.to_excel(writer)

    return df_new
