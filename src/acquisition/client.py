import configparser 
from spacetrack import SpaceTrackClient, like
import os 


# -- constantes REST (conservées pour référence, non utilisées avec le client)
uriBase             = "https://www.space-track.org"
requestLogin        = "/ajaxauth/login"
requestCmdAction    = "/basicspacedata/query"
requestOMMStarlink1 = "/class/gp/NORAD_CAT_ID/"
requestOMMStarlink2 = "/orderby/EPOCH/format/xml"

# -- Constantes de requetes
SAMPLES = 50
PERIOD = 30


def initClient(ini_path):
    # -- config 
    config = configparser.ConfigParser()
    config_path = ini_path or os.path.join(os.path.dirname(__file__), "SpaceTrack.ini")
    config.read(config_path)
    configUsr = config.get("configuration", "username")
    configPwd = config.get("configuration", "password")
    
    print("Connexion à SpaceTrack...")
    st = SpaceTrackClient(identity=configUsr, password=configPwd)
    print("Connecté.")
    return st
