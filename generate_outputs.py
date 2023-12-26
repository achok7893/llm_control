# -*- coding: utf-8 -*-
import pandas as pd
import copy
import os
import sys
import json
import logging
import time

os.system("pip install --upgrade openai")
os.system("pip install openpyxl")
os.system("pip install azure-storage-file-datalake azure-identity")

from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azureml.core import Workspace

params = json.load(open("./secrets.json","r"))
storage_account_name = params["storage_account_name"]
storage_account_key = params["storage_account_key"]
container_name = params["container_name"]
blob_name = "test_upload2.xlsx"
local_file_path = "test_upload.xlsx"
blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=storage_account_key)   
client = AzureOpenAI(
    api_key=params["openai_api_key"],  
    api_version="2023-12-01-preview",
    azure_endpoint =params["azure_endpoint"]
)


def get_current_time():
    """Get current time in string format"""
    import time
    strings = time.strftime("%Y,%m,%d,%H,%M,%S")
    t = strings.split(',')
    str_current_time = "".join([str(x) for x in t])
    return str_current_time


def set_logging_policy():
    "Set logging policy"

    current_time = get_current_time()
    log_path_file = f'./logs_{current_time}.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=log_path_file,
                        filemode='w')
    logger = logging.getLogger('logger')

    def exc_handler(exctype, value, tb):
        logger.exception(
            ''.join(traceback.format_exception(exctype, value, tb)))
    sys.excepthook = exc_handler
    
    return log_path_file


def get_cr(x):
    l = x.split("__________________")
    try:
        return l[1]
    except:
        return l[0]

def get_messages_for_fictive_cr(x):
    message = [{"role":"system","content":"Réponds à l'instruction."},
               {"role":"user","content":f"""
               ### INSTRUCTION : Génère un CR médical fictif dans le cadre de la chirurgie hépatique avec la même forme en t’inspirant du CR fourni mais en modifiant les dates, les noms propres, les mesures, les noms des villes, la structure familiale et les éventuelles professions pour qu’il soit fictif. 
               ### CR : {x}"""}]
    return copy.deepcopy(message)

def get_messages_summaries(x):
    message = [{"role":"system","content":"Réponds à l'instruction."},
               {"role":"user","content":f"""
               ### INSTRUCTION : Fournis dans un json {{"Résumé complet" : "<résumé>", "Résumé avec des omissions importantes": "<résumé incomplet", "Résumé complet mais avec des hallucinations": "<résumé avec hallucinations"}}
               ### CR : {x}"""}]
    return copy.deepcopy(message)

def get_ner_from_cr(x):
    l_ner = ['presence_chc','presence_hepatectomie','rad_splenomegalie_receveur','thrombose_porte','retour_porte','shunt','ascite','antecedent_infection_ascite','tips','traitement_chc','chimioembolisation','transplantation ','radiofrequence','radiotherapie','radioembolisation','sevrage_alcool','hepatite_B','hepatite_C','eradication_C','encephalopathie','hemorragie','antecedent_covid', 'hosptitalisation_covid']
    l_ner = " : 'Oui', 'Non', 'NA' \n".join(l_ner) +  ": 'Oui', 'Non', 'NA'"
    message = [{"role":"system","content":"Réponds à l'instruction."},
               {"role":"user","content":f"""
               ### Instruction :
               Uniquement à partir du CR fourni il faudra me rengoyer un json rempli pour répondre aux questions suivantes ("NA" veut dire qu'on ne peut pas dire si c'est présent ou absent à partir du CR) : 
               {l_ner}
               "biologie" : {{"<valeur mesurée>": [<valeur en float>, "unité en str"], ....}}
               "date_hospitalisation" : "dd/mm/yyyy" ou "NA"
               "date_transplantation" : "dd/mm/yyyy" ou "NA"
               "date_chimiobolisation" : "dd/mm/yyyy" ou "NA"
               ### CR : {x}"""}]
    return copy.deepcopy(message)  

def process_one_input(x):
    res_ = {}
    chat_completion = client.chat.completions.create(
        model="gpt4_32k", # model = "deployment_name".
        messages=get_messages_for_fictive_cr(x)
    )
    res_["fictive_cr"] = chat_completion.choices[0].message.content
    
    chat_completion = client.chat.completions.create(
        model="gpt4_32k", # model = "deployment_name".
        messages=get_messages_summaries(res_["fictive_cr"])
    )
    res_["summaries"] = copy.deepcopy(chat_completion.choices[0].message.content)

    chat_completion = client.chat.completions.create(
        model="gpt4_32k", # model = "deployment_name".
        messages=get_ner_from_cr(x)
    )
    res_["ner_from_cr"] = copy.deepcopy(chat_completion.choices[0].message.content)

    return copy.deepcopy(res_)

if __name__=="__main__":
    try:
        log_path_file = set_logging_policy()

        logging.info("Reading input dataset")
        df_raw = pd.read_csv("./CR_TH.csv", sep=";")
        logging.info("processing input column")
        df_raw.loc[:,"CR_raw"] = df_raw.loc[:,"contenu"].apply(get_cr)

        l_resultats = []
        container_client = blob_service_client.get_container_client(container_name)

        for i in range(df_raw.shape[0]):
            logging.info(f"Iteration {i}")
            for j in range(2):
                logging.info(f"SubIteration {j}")
                try:
                    l_resultats.append(process_one_input(df_raw.loc[:,"CR_raw"].iloc[i]))
                except:
                    try:
                        logging.info(f"retry after 60 s")
                        time.sleep(60)
                        l_resultats.append(process_one_input(df_raw.loc[:,"CR_raw"].iloc[i]))
                    except:
                        logging.info(f"retry after 300 s")
                        time.spleep(300)
                        l_resultats.append(process_one_input(df_raw.loc[:,"CR_raw"].iloc[i]))
                logging.info("response obtained")
                pd.DataFrame(l_resultats).to_pickle("./fictifs_cr.pickle")
                try:
                    blob_client = container_client.get_blob_client("fictifs_cr.pickle")
                    blob_client.delete_blob()
                    logging.info("Excel File deleted")
                except:
                    logging.info("No file to delete")        
                            # Upload the file to Azure Blob Storage
                with open("fictifs_cr.pickle", "rb") as data:
                    container_client.upload_blob(name="fictifs_cr.pickle", data=data)
                    logging.info("file uploaded")
                logging.info(f"File '{local_file_path}' uploaded to Blob Storage.")
                
                with open(log_path_file, "rb") as data:
                    container_client.upload_blob(name=log_path_file, data=data)
                    logging.info("file uploaded")
    except:
        with open(log_path_file, "rb") as data:
            container_client.upload_blob(name=log_path_file, data=data)
            logging.info("file uploaded")
