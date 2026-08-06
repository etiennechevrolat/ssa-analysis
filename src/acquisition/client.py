import configparser 
from spacetrack import SpaceTrackClient
import os 


def initClient(ini_path):
    # -- config 
    config = configparser.ConfigParser()
    config_path = ini_path or os.path.join(os.path.dirname(__file__), "SpaceTrack.ini")
    config.read(config_path)
    configUsr = config.get("configuration", "username")
    configPwd = config.get("configuration", "password")
    
    print("Connexion à SpaceTrack...")
    st = SpaceTrackClient(identity=configUsr, password=configPwd)
    print("Client initialisé.")
    return st
