import pandas as pd
import __init__ as init
import json

def SelectMessage(message_idx):

    if message_idx == 6:
         return 'MESSAGE: Feedback'
    
    if message_idx == 5: # no messages available for [3,1]
         return 'MESSAGE: [3,1]'

    message_type = getCategory(message_idx) # e.g., [1,1]
    df = makeData()

    message = df[(df['phase_codes'] == message_type[0]) & (df['determinant_codes'] == message_type[1]-1) ] #other = -1, selfeff = 0
    message = message.sample()['message']
    
    return 'MESSAGE: ', message.values[0]



def getCategory(message_idx):
    with open(init.dict +'message_descriptors.json', 'r', encoding='utf-8') as file_handle:
            all_messages = json.load(file_handle)
    
    mes_descr = None
    for m in all_messages:
            if m['ID'] == message_idx:
                mes_descr = m['descr']  # phase, determinant / feedback 
                break
    return mes_descr

def makeData():
      
    message_data = 'C:\\Users\\sebas\\Desktop\\messages.csv'
    df_mssg = pd.read_csv(message_data, encoding='unicode_escape',sep=';')

    df_mssg.phases = pd.Categorical(df_mssg.phases, categories=['ALLPHASES', 'INITIATION', 'ACTION', 'MAINTEN'], ordered=True)
    df_mssg['phase_codes'] = df_mssg.phases.cat.codes #  0, 1, 2, 3

    df_mssg.determinant = pd.Categorical(df_mssg.determinant, categories = ['SELFEFF'], ordered=True)
    df_mssg['determinant_codes'] = df_mssg.determinant.cat.codes # -1, 0 

    
    df_mssg = df_mssg[['BehaviorGoal', 'message', 'phase_codes', 'determinant_codes']]
    df_mssg = df_mssg.drop(df_mssg[df_mssg['BehaviorGoal'] == 'DIET'].index)

    #print(df_mssg.head())


    return df_mssg


#print(SelectMessage(1))