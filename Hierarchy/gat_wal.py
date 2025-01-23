import numpy as np
import pandas as pd

import os
import json
import torch
import numpy as np
import random

from torch.utils import data

import pandas as pd
import csv


import random


import pandas as pd
import torch
from model.model import TranHGAT
from torch.utils import data
from model.dataset import Dataset
from train import initialize_and_train
import csv
import os

import random

import nltk
nltk.download('words')
from nltk.corpus import words
word_list = words.words()


CUDA = 0
N_epoch = 100



def dynamic_convert_csv_to_txt(input_csv, output_txt):
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_txt, mode='w', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        for row in reader:
            parts = {'left': '', 'right': ''}
            for col_name, value in row.items():
                if '_' in col_name:
                    prefix, attribute = col_name.rsplit('_', 1)
                    parts[prefix] += f"COL {attribute} VAL {value} "
            
            left_part = parts.get('left', '').strip()
            right_part = parts.get('right', '').strip()
            label = row.get('label', '').strip()
            
            outfile.write(f"{left_part} \t{right_part} \t{label}\n")


#print(0)







import numpy as np
import pandas as pd
import time

brand_hierarchy = {
    "Electronics": {
        "Computers": ["acer", "alienware", "apple", "asus", "dell", "lenovo", "microsoft", "a-data", "a4tech", "accell", "avid", "azio", "ibm", "4inkjets", "acme", "acme made", "acp", "acp-ep memory", "agama", "aoc", "apc", "arclyte technologies inc.", "atlona", "atrend", "aurora", "avf", "basyx", "bell o", "bling software", "blurex", "bracketron", "buffalo technology", "built ny"],
        "Components": ["3m", "corsair", "kingston", "seagate", "sandisk", "western digital", "dane", "dane-elec", "diamond multimedia", "bits limited", "coolmax", "memory upgrades", "patriot", "buffalo", "centon", "chief", "chief manufacturing", "clickfree", "cobra", "comprehensive", "comprehensive video", "coolguard", "cp tech", "cp technology", "crown", "edge", "eforcity", "ep memory", "eveready", "focus"],
        "Mobile Devices": ["htc", "jabra", "samsung", "sony", "nokia", "apple", "blackberry", "motorola", "mybat", "coby", "lg", "vtech"],
        "Audio & Video": ["audio-technica", "bose", "jvc", "pioneer", "plantronics", "sony", "audiovox", "audiosource", "axis", "axis communications", "boss", "creative", "crosley", "diamond", "jaton", "razer", "sennheiser", "sharp", "skullcandy", "creative labs", "hercules", "ihome", "elite", "matic", "clarion mobile electronics", "dj tech", "dj-tech", "fanatics", "philips", "rca", "sirius", "sirius satellite radio"],
        "Networking": ["cisco", "d-link", "linksys", "netgear", "tp-link", "zyxel", "actiontec", "trendnet", "trendware", "amp", "amped wireless", "link depot", "syba", "syba multimedia", "netopia", "u.s. robotics"],
        "Storage Devices": {
            "Internal Storage": ["seagate", "western digital", "toshiba", "crucial", "ocz", "ocz technologies", "quantum", "transcend", "edge tech"],
            "External Storage": ["sandisk", "kingston", "lacie", "buffalo", "cavalry", "buslink", "apricorn", "iomega", "pny"]
        }
    },
    "Office Supplies": {
        "Paper Products": ["hammermill", "smead", "avery", "cardinal", "pendaflex", "boise", "domtar", "pacon", "pacon products", "boise cascade", "tops business forms"],
        "Writing Instruments": ["bic", "sharpie", "pilot", "papermate", "uniball", "prismacolor", "expo", "tops", "pentel", "pentel of america ltd.", "stabilo"],
        "Organization": ["wilson jones", "officemate", "post-it", "swingline", "bankers box", "buddy products", "c-line", "smead manufacturing company", "rolodex", "globe weis", "ghent", "safco products"],
        "Other Office Supplies": ["acco", "buddy products", "dixie", "rubbermaid", "samsill", "bond street", "bond street ltd.", "elmer s", "exact", "staples", "at t", "avery personal creations", "paperpro", "redi-tag"]
    },
    "Accessories": {
        "Computer Accessories": ["accessory power", "belkin", "case logic", "kensington", "targus", "brenthaven", "cables to go", "cables unlimited", "iogear", "zagg", "otterbox", "incipio", "griffin", "vantec", "logitech", "cooler master", "coolermaster", "pluggable technologies", "siig", "planar", "planar systems", "plugable technologies"],
        "Mobile Accessories": ["zagg", "otterbox", "speck", "incipio", "griffin", "blueant", "iluv", "amzer", "just mobile", "bluelounge", "bluelounge design", "roadwired"],
        "Photography Accessories": ["lowepro", "moshi", "manfrotto", "tamron", "sigma", "ape case", "dolica", "tiffen", "tenba", "zeikos", "arkon", "arrowmounts", "bower", "platt", "platt cases", "sima", "golla", "humminbird"]
    },
    "Software": {
        "Operating Systems": ["microsoft"],
        "Creative Tools": ["adobe", "encore software", "acd", "acd systems", "avid technology", "roxio", "serif", "the learning company"],
        "Security": ["symantec", "webroot", "mcafee", "kaspersky"],
        "Utilities": ["avg", "norton", "ccleaner", "winrar", "plustek", "neosafe", "advanced system care", "iosafe"]
    },
    "Consumer Electronics": {
        "Audio": ["bose", "audio-technica", "plantronics", "skullcandy", "sony", "aztec", "soundstorm", "fellowes", "cyber acoustics", "ultimate ears"]
    },
    "Printing & Imaging": {
        "Printers": ["brother", "canon", "epson", "hp", "xerox", "oki", "lexmark", "dymo"],
        "Cameras": ["nikon", "olympus", "panasonic", "sony", "vivitar", "fujifilm", "kodak", "polaroid", "fuji", "garmin"],
        "Projectors": ["optoma", "benq", "epson", "nec", "hitachi", "elite screens", "viewsonic", "sunpak"]
    },
    "Home Appliances": {
        "Small Appliances": ["black & decker", "dyson", "kitchenaid", "cuisinart", "allied", "haier", "emerson", "magna visual", "keurig"]
    },
    "Entertainment": {
        "Characters": ["disney", "hello kitty"],
        "Media": ["sirius xm", "spotify", "pandora", "ilive", "flip video", "eye-fi", "favi entertainment", "vizio", "sirius", "sirius satellite radio"]
    },
    "Miscellaneous": {
        "General": ["generic", "unknown", "3m #", "blue crane digital", "black n  red", "green onions supply", "u.s. brown bear", "universal", "americopy", "dealz4real", "egpchecks", "add on", "sage", "sumas"],
        "Multi-Category": ["3m", "amazonbasics", "anker", "eco-products", "advantus", "acs", "casio", "ford"]
    },
    "Uncategorized": ["accessorieszone", "agf", "action", "adesso", "alama", "aluratek", "antec", "balt", "best rite", "booq", "brainydeal", "bti", "bushnell", "compatible", "digital concepts", "dreamgear", "duracell", "energizer", "evga", "ifixit", "joby", "kensmart", "lowrance", "macsense", "marware", "mivizu", "msi", "mustang", "namo", "nuance", "penpower", "quartet", "storex", "think outside", "treque", "tribeca", "tripp lite", "trodat", "v7", "vaultz", "writeright", "zalman", "clarion", "clarion software", "simplism", "cocoon", "cocoon innovations", "compucessory", "concord", "csdc", "cta", "cta digital", "curtis", "cyberpower", "da-lite", "da-lite screen", "defender", "deluxe", "diaper dude", "digipower", "digital innovations", "digital peripheral solutions", "directed electronics", "discgear", "dp audio video", "dp video", "draper", "eaton", "elago", "electrified", "ematic", "emerge tech", "encore", "endust", "eveready", "fantom", "franklin", "franklin electronics", "gator", "gator cases", "gbc", "ge", "gear head", "general electric", "genius", "genuine phillips", "global marketing", "global marketing partners", "gn netcom", "goodhope bags", "gpx", "grade-a", "greatshield", "green", "griffin technology", "grt", "guardian", "hewcpn", "hon", "hqrp", "human toolz", "humantoolz", "i concepts", "i-tec", "i-tec electronics", "i.sound", "ifrogz", "igo", "imation", "inland", "inland pro", "innergie", "innovera", "iosafe inc", "iris", "iriver", "itw", "itw dymon", "ivina", "jawbone", "jwin", "kanguru", "kanguru solutions", "karendeals", "kenwood", "keyspan", "keytronicems", "kimberly-clark", "kimberly-clark professional", "kinamax", "kingsmart", "kingwin", "kiq", "konica-minolta", "koss", "labtec", "lanthem", "lathem", "lexar", "lexar media", "lifeworks", "lorex", "lumiere l.a.", "lumiere la", "luxor", "macally", "maccase", "mace", "mace security", "mach speed", "magnavox", "manhattan", "master caster", "maxell", "mayline", "mead", "memorex", "memtek", "mercury", "mercury luggage", "merkury", "merkury innovations", "metra", "micro innovations", "micronet", "middle atlantic", "midland", "mionix", "mmf", "mobile edge", "mohawk", "molex", "motion systems", "mukii", "mygear products", "nan", "national", "national products ltd.", "navgear", "neat receipts", "neatreceipts", "next web sales", "nextware", "night owl", "night owl optics", "norazza", "nxg", "nxg technology", "nzxt", "office star", "oki data", "omnimount", "orbital", "oxford", "p3 international", "p3 international corporation", "paper mate", "pc treasures", "peak", "peerless", "pelican", "pelican storm", "pelouze", "pentel of america ltd.", "performance plus", "planar", "planar systems", "plugable technologies", "pm company", "power mat", "power mate", "powermat", "premiertek", "primera technology", "pyle", "pyramid", "q-see", "quality park", "raptor", "raptor-gaming", "rayovac", "read right", "rim", "riteav", "roocase", "royal", "royal consumer", "s j paper", "sabrent", "saitek", "sakar", "sanford", "sanrio", "sanus", "sanyo", "sapphire", "scosche", "scotch", "seal shield", "seiko", "seiko instruments", "sentry", "sgp", "shopforbattery", "simplism japan", "siskiyou", "skooba design", "slappa", "slik", "smartbuy", "smk", "solidtek", "solo", "sonnet technologies", "sonnet technologies inc", "sparco", "spectrum brands", "srs", "srs labs", "stanley", "stanley bostitch", "startech", "startech.com", "startech.com usa llp", "steelseries", "storm", "sumas media", "sumdex", "sunvalleytek", "super talent", "svat", "svat electronics", "swann", "sylvania", "t-mobile", "talk works", "tandberg", "team pro mark", "team promark", "team promark llc", "tektronix", "ten", "ten one design", "terk", "the joy factory", "thermaltake", "tomtom", "tp link", "tracfone", "tvtimedirect", "ultralast", "uniden", "universal products", "us brown bear", "usrobotics", "veho", "verbatim", "victor", "victory", "victory multimedia", "vipertek", "visioneer", "visiontek", "vistablet", "visual land", "vonnic inc", "wacom", "wacom tech corp.", "wausau", "wausau paper", "weyerhauser", "whistler", "whistler radar", "wincraft", "wintec", "x-acto", "x16-81686-03", "xantech", "xfx", "xgear", "xo vision", "xtrememac", "zalman usa inc", "zax", "zebra", "zoom", "zoom telephonics", "zotac", "zune"],
    'UNK':["aiptek", "alera", "aleratec", "alkaline", "allsop", "amp energy", "ampad", "arclyte technologies inc. .", "atrend-bbox", "bell  o", "bravo", "bravo view", "buddy", "can-c", "channel sources", "clover", "clover electronics", "cms", "curo7", "pentel of america ltd. ."]
}
category_hierarchy = {
    "Electronics": {
        "Audio": {
            "Cables": ["audio cables", "speaker cables", "rca cables", "video cables", "cables interconnects", "power cables"],
            "Mobile Phones": ["prepaid wireless phones", "unlocked cell phones", "smartphones", "phones", "feature phones", "corded telephones", "cordless telephones", "phones with plans", "unlocked phones", "no-contract phones"],
            "Car Audio": ["audio car mounts", "car audio video", "car stereos", "bluetooth car kits", "auxiliary input adapters", "audio-video kits", "car kits", "radio antennas", "equalizers"],
            "Speakers": ["stereos/audio", "coaxial speakers", "speakers", "subwoofers", "component subwoofers", "speaker systems", "speaker parts components", "subwoofer boxes and enclosures", "boomboxes"],
            "Accessories": ["amplifier installation", "amplifier wiring kits", "speaker installation", "mp3 accessories", "speaker connectors", "bluetooth headsets", "universal fm cassette adapters", "mp3 player accessories", "mp3 player cables adapters", "headsets", "wired headsets", "headsets microphones", "microphones accessories", "touch screen tablet accessories"],
            "General": ["audio video accessories", "computers accessories", "phone accessories", "electronics-inflexible kit", "satellite radios", "satellite radio", "antennas", "handheld portable satellite radios", "plug play satellite radios", "component video", "stereos", "answering devices", "caller id displays", "corded-cordless combo telephones"]
        },
        "Cameras": {
            "Types": ["digital cameras", "film cameras", "hidden cameras", "game cameras", "bullet cameras", "dome cameras", "simulated cameras", "camcorders", "dslr cameras", "digital slr cameras", "digital slr camera bundles", "point shoot digital cameras", "point shoot digital camera bundles", "film"],
            "Accessories": ["camera batteries", "lens accessories", "tripods", "camera and camcorder accessories", "camera lenses", "binocular accessories", "filters", "camera bags", "camcorder batteries", "photo video design", "digital camera accessories", "flashes", "complete tripod units", "professional video accessories", "webcams", "external floppy drives", "faceplates", "screen filters"],
            "General": ["camera photo", "binoculars"]
        },
        "Computing": {
            "Components": ["motherboards", "graphics cards", "hard drives", "internal hard drives", "external hard drives", "internal optical drives", "internal sound cards", "computer components", "memory", "memory cards", "usb port cards", "network cards", "case fans", "video capture cards", "optical drives", "hard drive enclosures", "firewire port cards", "internal dvd drives", "i o port cards", "floppy diskettes"],
            "Accessories": ["keyboard mouse combos", "mouse pads", "computers", "laptop computers", "monitor arms stands", "computer cases", "cooling pads", "usb cables", "power supplies", "computer accessories", "memory card readers", "memory card adapters", "monitors", "mice", "gaming mice", "modems", "scsi port cards", "computer cable adapters", "computer monitor mounts", "computer speakers", "cases", "keyboards", "laptops", "netbooks", "trackballs", "touch pads", "headphones", "headphone accessories", "desktops", "keyboards mice input devices", "keyboards styluses", "laptop netbook computer accessories", "tablet accessories", "docking stations", "graphics tablets", "monitor accessories", "mounts", "tablets", "ipod"],
            "Software": ["operating systems", "productivity software", "security software", "graphic design software", "development tools", "software"]
        },
        "Networking": {
            "Devices": ["routers", "walkie-talkie/frs", "network adapters", "wireless access points", "modems", "powerline network adapters", "print servers", "switches", "network attached storage", "frs two-way radios", "gmrs-frs two-way radios", "two-way radios accessories", "kvm switches", "usb network adapters", "networking products"],
            "Cables": ["ethernet cables", "usb cables", "data cables", "firewire cables", "scsi cables", "dvi cables", "hdmi cables", "charger cables", "chargers cables"]
        },
        "Televisions": {
            "TVs": ["tvs", "television", "televisions video", "overhead video"],
            "Accessories": ["tv mounts", "projection screens", "television stands entertainment centers", "video projectors", "video projector accessories", "remote controls", "remote-control extenders", "tv accessories", "video glasses", "projector accessories", "projector mounts", "video converters", "video receiving & installation", "video transmission systems", "controllers"]
        },
        "General": ["electronics - general", "networking", "office electronics", "office electronics accessories", "hubs", "components", "power strips", "cleaning repair", "distribution panels", "electronics", "electronics : flat panel tv", "audio-video shelving", "impact dot matrix printer ribbons", "electrical", "distribution", "surge protectors", "uninterrupted power supply ups", "laser printers", "printing"]
    },
    "Office Supplies": {
        "Paper Products": ["laminating supplies", "business cards", "memo scratch pads", "labels stickers", "postcards", "wide-format paper", "business paper products", "colored paper", "file folder labels", "roll paper", "shipping labels", "address labels", "continuous-form labels", "all-purpose labels", "photo paper", "paper"],
        "Furnitures": ["furniture", "desks", "chairs", "file cabinets", "office furniture lighting", "hanging folders interior folders", "footrests", "stands", "wrist rests"],
        "Electronics": ["printers", "scanners", "shredders", "fax machines", "inkjet all-in-one printers", "laser all-in-one printers", "digital security recorders", "all-in-one printers", "projection screens", "desktop staplers", "telephone accessories", "scanner accessories", "label makers", "postal scales"],
        "Printer Accessories": ["printer accessories", "inkjet printer ink", "laser printer toner", "printer ink toner", "printer labels laser inkjet", "printer staples", "printer roll holders", "printer transfer rollers", "printer transfer units", "inkjet printer paper"],
        "Stationery": ["self-stick notes", "lamps", "pen holders", "stationery & office machinery", "labeling tapes", "binder index dividers", "looseleaf binder paper", "labels stickers", "badge holders", "storage presentation materials", "document creation", "view binders", "english dictionaries", "self-stick note pad holders", "flowcharts", "portfolios", "presentation pointers", "presentation remotes", "index dividers"]
    },
    "Automotive": {
        "Electronics": ["dash mounting kits", "radar detectors", "gps", "handheld gps", "vehicle gps", "portable vehicle gps", "in-dash navigation", "audio-video kits", "gps system accessories", "car electronics"],
        "Accessories": ["car chargers", "vehicle mounts", "back seat cushions", "car cradles", "dash mounting kits", "wiring harnesses", "auxiliary input adapters", "cb radios", "antitheft", "car electronics accessories"],
        "General": ["automotive - general"]
    },
    "Batteries": {
        "Types": ["12v", "6v", "9v", "aa", "aaa", "coin button cell", "batteries"],
        "Accessories": ["battery chargers", "charger cables", "chargers adapters", "household batteries", "batteries chargers accessories", "desktop chargers", "ac adapters", "chargers"]
    },
    "Media": {
        "Storage": ["storage", "cd-r discs", "dvd r discs", "usb flash drives", "usb drives", "external data storage", "data cartridges", "dvd rw discs", "bd-r discs", "blank media", "cd-rw discs", "dvd-ram discs", "dlt cleaning cartridges", "media storage organization", "disc jewel cases", "disc storage wallets", "dvd-r discs"],
        "Devices": ["blu-ray disc players", "mp3 players", "portable dvd players", "dvd players", "dvd accessories", "upconverting dvd players", "disc players recorders", "mp3", "portable audio", "cd players", "cd-mp3 players", "digital media devices", "changers"]
    },
    "Home & Security": {
        "Home Theater": ["dvd home theater", "home theater systems", "home theater", "stereo amplifiers", "home audio theater", "home audio crossovers", "turntables"],
        "Security": ["home care", "home security systems", "surveillance cameras", "security surveillance", "security sensors alarms", "complete surveillance systems", "hidden cameras", "household sensors alarms", "simulated cameras", "security monitors displays", "surveillance video recorders"],
        "General": ["house wares"]
    },
    "Accessories": {
        "General": ["bags cases", "covers skins", "armbands", "cases sleeves", "handbags", "camera bags", "accessories", "accessories supplies", "accessories apparel", "accessory kits", "cases bags", "hard drive bags", "projector bags cases", "covers", "armbands", "styli", "screen protectors", "screen protector foils", "wall chargers", "laser pointers"]
    },
    "Tools": {
        "General": ["toolkits", "power strips", "cable security devices", "cleaning repair", "binding machines", "binding machine supplies", "adapters", "connectors adapters", "adapter rings", "distribution panels", "wire management", "power adapters", "power converters", "power ground cable", "power inverters", "power-cable terminals", "selector boxes", "switchers"]
    },
    "Photography": {
        "Cameras": ["point shoot digital cameras", "dslr cameras", "camcorders", "digital slr cameras", "digital slr camera bundles", "point shoot digital camera bundles"],
        "Accessories": ["filters", "camera bags", "tripods", "camera batteries", "camera lenses", "flashes", "lens accessories", "complete tripod units", "professional video accessories"],
        "General": ["photography - general", "photo editing", "media storage organization"]
    },
    "Other": {
        "General": ["scientific", "fan shop", "rugs", "toys - games", "garden - general", "sports outdoor gps", "hiking gps units", "boating gps units chartplotters", "fish finder", "fishfinders", "fishing and boating", "nan", "personal care", "other office equipment", "outlet plates", "wall plates connectors", "radios", "basic", "bundles", "c", "d", "hardware", "lighting", "pda handheld accessories"]
    }
}




def make_mask(df,frac):
    L = len(COL)*len(df)
    L_disturb = int(np.ceil(L*frac))
    disturb_mask = np.zeros((len(df),len(COL)))
    total_elements = disturb_mask.size
    random_indices = np.random.choice(total_elements, L_disturb, replace=False)
    disturb_mask.flat[random_indices] = 1
    return disturb_mask 


def find_parents(hierarchy, target, parents=None):
    if parents is None:
        parents = []

    for key, value in hierarchy.items():
        if key == target:  # If the target itself is a key
            return parents + [key]
        if isinstance(value, dict):  # If value is a nested dictionary
            result = find_parents(value, target, parents + [key])
            if result != -1:
                return result
        elif isinstance(value, list) and target in value:  # If value is a list containing the target
            return parents + [key]

    return -1  # Return -1 if the target is not found


def disturb_func(col,value):
    if col =='brand':
        
        parents = find_parents(brand_hierarchy, str(value).lower().strip())
        if parents == -1: print(value)
        new_seed = int(time.time())
        np.random.seed(new_seed)
        np.random.shuffle(parents)
        new_velue = parents[0]
        

    elif col == 'category':
        parents = find_parents(category_hierarchy, str(value).lower().strip())
        if parents == -1: print(value)
        new_seed = int(time.time())
        np.random.seed(new_seed)
        np.random.shuffle(parents)
        new_velue = parents[0]

    elif col =='price':
        if str(value) =='nan':
            new_velue = 'other'

        else:
                        
            data = [float(value)]
            number_hierarchy = {
                "Ranges": {
                    "Very Low": {
                        "0-0.49": [n for n in data if 0 <= n < 0.5],
                        "0.5-0.99": [n for n in data if 0.5 <= n < 1]
                    },
                    "Low": {
                        "1-4.99": [n for n in data if 1 <= n < 5],
                        "5-9.99": [n for n in data if 5 <= n < 10]
                    },
                    "Moderate": {
                        "10-49.99": [n for n in data if 10 <= n < 50],
                        "50-99.99": [n for n in data if 50 <= n < 100]
                    },
                    "High": {
                        "100-499.99": {
                            "100-199.99": [n for n in data if 100 <= n < 200],
                            "200-299.99": [n for n in data if 200 <= n < 300],
                            "300-399.99": [n for n in data if 300 <= n < 400],
                            "400-499.99": [n for n in data if 400 <= n < 500]
                        },
                        "500-999.99": {
                            "500-749.99": [n for n in data if 500 <= n < 750],
                            "750-999.99": [n for n in data if 750 <= n < 1000]
                        }
                    },
                    "Very High": {
                        "1000-4999.99": {
                            "1000-1999.99": [n for n in data if 1000 <= n < 2000],
                            "2000-2999.99": [n for n in data if 2000 <= n < 3000],
                            "3000-3999.99": [n for n in data if 3000 <= n < 4000],
                            "4000-4999.99": [n for n in data if 4000 <= n < 5000]
                        },
                        "5000 and above": {
                            "5000-9999.99": [n for n in data if 5000 <= n < 10000],
                            "10000 and above": [n for n in data if n >= 10000]
                        }
                    }
                },
                
            }
            if find_parents(number_hierarchy, float(value)) ==-1:
                print(value, type(value))
            parents = find_parents(number_hierarchy, float(value))[1:]

            new_seed = int(time.time())
            np.random.seed(new_seed)
            np.random.shuffle(parents)
            new_velue = parents[0]
        
    return new_velue



# /home/mmoslem3/DATA_VLDB/Fodors-Zagat/

task = 'Walmart-Amazon'
COL = ['category','brand','price']


import os
import random
for frac in [0,0.1,0.2,0.3,0.4,0.5]:
    for rep in range(1,16):
        # df = pd.read_csv(f'data/{task}/test.csv')
        df = pd.read_csv("/home/mmoslem3/DATA_VLDB/" + task  +'/test.csv')
        mask_left = make_mask(df,frac)
        mask_right = make_mask(df,frac)
        df['left_price'] = df['left_price'].astype(str)
        df['right_price'] = df['right_price'].astype(str)
        for i,col in enumerate(COL):
            df.loc[mask_left[:,i] == 1, 'left_'+col] = df.loc[mask_left[:,i] == 1, 'left_'+col].apply(lambda x: disturb_func(col, x))
            df.loc[mask_right[:,i] == 1, 'right_'+col] = df.loc[mask_right[:,i] == 1, 'right_'+col].apply(lambda x: disturb_func(col, x))




        model_path = 'VLDB_best_model_'+task+'.pth'
        dataset_dir = "/home/mmoslem3/DATA_VLDB/" + task  + "/"
        embedding_dir = '/home/mmoslem3/Hiermathcer/embedding' 
        df.to_csv(dataset_dir+'test_noisyWord.csv',index=False)





        data_dir = dataset_dir

        train_path = "valid.csv"
        valid_path = "valid.csv"
        test_path = "test.csv"

        for i in [valid_path, train_path]:
            if os.path.exists(os.path.join(data_dir, i.replace('csv','txt'))): continue
            dynamic_convert_csv_to_txt(os.path.join(data_dir, i), os.path.join(data_dir, i.replace('csv','txt')))



        i = "test_noisyWord.csv"
        dynamic_convert_csv_to_txt(os.path.join(data_dir, i), os.path.join(data_dir, i.replace('csv','txt')))


        run_id = 0
        batch_size =32

        max_len = 256
        lr = 1e-5
        n_epochs = N_epoch
        finetuning = True
        save_model = True


        model_path = 'vldb_saved_model_'+task +'/'
        lm_path = None
        split = True
        lm = 'bert'





        args = {
            "batch_size": batch_size,
            "lr": lr,
            "n_epochs": n_epochs,
            "finetuning": finetuning,
            "save_model": save_model,
            "model_path": model_path,
            "lm_path": lm_path,
            "lm": lm}







        trainset = os.path.join(data_dir,train_path.replace('csv','txt'))
        validset = os.path.join(data_dir,valid_path.replace('csv','txt'))
        testset = os.path.join(data_dir,test_path.replace('csv','txt'))


        # load train/dev/test sets
        train_dataset = Dataset(trainset, ["0", "1"], lm=lm, lm_path=lm_path, max_len=max_len, split=split)
        valid_dataset = Dataset(validset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)
        test_dataset = Dataset(testset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)




        device = 'cuda:'+str(CUDA) if torch.cuda.is_available() else 'cpu'
        model = TranHGAT( train_dataset.get_attr_num(), device, finetuning, lm=lm, lm_path=lm_path)
        model.load_state_dict(torch.load(model_path+'/'+'model.pt', map_location= device))
        model = model.to(device)
        model.eval()





        torch.cuda.empty_cache()
        frac2 = str(int(frac*100))
        if len(frac2) == 1: frac2 = frac2 +'0'



        test_path = 'test_noisyWord.csv'







        

        testset = os.path.join(data_dir,test_path.replace('csv','txt'))
        test_dataset = Dataset(testset, ["0", "1"], lm=lm, lm_path=lm_path, split=split)





        all_probs = []
        all_y =[]

        test_iter = data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False, num_workers=0, collate_fn=Dataset.pad)
        with torch.no_grad():
            for i, batch in enumerate(test_iter):
                _, x, y, _, masks = batch
                logits, y1, y_hat = model(x.to(device), y.to(device), masks.to(device))
                logits = logits.view(-1, logits.shape[-1])
                probs = logits.softmax(dim=1)[:, 1]
                all_probs += probs.cpu().numpy().tolist()
                all_y += y1.cpu().numpy().tolist()
                


        y_true = all_y
        y_score = all_probs
        df = pd.concat([pd.DataFrame(y_true, columns=['y_true']), pd.DataFrame(y_score, columns=['y_score'])], axis=1)
        df.to_csv('/home/mmoslem3/RES_hierarch/HierGATWalmart/HG_score_'+task + '_'+frac2+'_'+str(rep)+'.csv', index=False)  
        # print("test_"+frac+'_'+str(rep)+".csv")
    # os.remove(data_dir+'/'+'train.txt')
    # os.remove(data_dir+'/'+'valid.txt')




