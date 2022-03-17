import os
import sys
sys.path.append(os.getcwd() + '/data')
sys.path.append(os.getcwd() + '/tools')
sys.path.append(os.getcwd() + '/source')


from tools.save_data_to_json import get_and_save

get_and_save(if_do_embedding=True, stanford_path='stanford-corenlp-4.3.1')