import math
import os
from decimal import Decimal, ROUND_HALF_DOWN

import pandas as pd
import numpy as np

class pre_process:
    def __init__(self, output_dir = '../../dataset'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.brand_mapping = {
            'landrover': 'land rover', 'land rover': 'land rover',
            'mercedes': 'mercedes-benz', 'mercedes-b': 'mercedes-benz',
            'vw': 'volkswagen', 'volkswagen': 'volkswagen',
            'gmc truck': 'gmc', 'gmc': 'gmc',
            'ford truck': 'ford', 'ford tk': 'ford', 'ford': 'ford',
            'dodge tk': 'dodge', 'dodge': 'dodge',
            'chev truck': 'chevrolet', 'chevrolet': 'chevrolet',
            'hyundai tk': 'hyundai', 'hyundai': 'hyundai',
            'mazda tk': 'mazda', 'mazda': 'mazda'
        }

    def categorize_trim(self, trim):
        if trim is None or (isinstance(trim, float) and math.isnan(trim)):
            return 'other'

        t = str(trim).lower().strip()

        categories = {
            'special edition': [
                'se', 'limited', 'lt', 'le', 'sel', 'ltz', '1lt', 'lt fleet', 'sle', '2lt', 'es', 'lt1',
                '2.5i premium pzev', 'unlimited sahara', '3,5 se', 'sle-1', 'king ranch', 'wolfsburg edition pzev',
                'se v6', 'special edition', 'zx4 se', 'sle-2', 'sle1', '2ss', 'ultimate', 'dx', 'el limited',
                'le v6', 'signature', 'se fleet', 'tdi', 'heat', 'lt3', '2.5i limited', 'sle 1500', '1500 sle',
                'wolfsburg edition', 'passion coupe', 'se1', 'sel plus'
            ],
            'touring': ['touring', 'slt', 'i touring', 'slt-1', 'slt-2', 'touring-l', 'grand touring', 'custom',
                        'lt 1500', 'slt 1500', 'i grand touring', 's grand touring', 'gtp'],
            'luxury': ['lx', 'xlt', 'gls', 'ex', 'ex-l', 'lariat', 'titanium', 'luxury', 'premium', 'xle', '+',
                       'denali', 'cxl', 'xe', '2.0 t premium quattro', 'laramie', 'platinum', 'technology package',
                       'premier', 'deluxe', 'cargo vn xlt', 'lx-p', 'gxe', 'gls 1.8t', 'glk350', 'gl450', 'gle',
                       'lx-s'],
            'sport': ['ls', 's', '2,5 s', 'sport', 'sv', '3,5 sv', 'sl', 'ls fleet', 'gt', '3,5 s', 'i sport', 'xls',
                      'sr5', '1,8 s', 'st', '3.2', '1500 ls', 'gs', '2,0 sr', '3,5 sl', '2.0t', 'sx',
                      'c300 sport 4matic', '2,0 s', 'c300 sport', 'stx', '1,6 s plus', '3.0si', 'sr', 'gt premium',
                      's pzev', 'v8', '1,8 sl', 'performance', 'supercharged', 's sport', 'turbo', 'prerunner v6',
                      '1,6 s', 'quattro', 'ls 3500', '3.5 sr', '2,0 sl', '3.2 quattro', 'sl500', 'gts', 'c350 sport',
                      'xle v6', 'sl 550', 'ex v6', 'sport pzev', 'r350', 'sl2', 'sle v6'],
            'base': ['base', 'sxt', 'xl', 'laredo', 'se pzev', 'r/t', 'hybrid', 'ce', 'american value', 'sxt fleet',
                     'cx', 'ltz fleet', 'tdi', 'crew', 'ltz 1500', 'rt', 'express', 'mainstreet', 'overland', 'eco',
                     'standard', 'fx35', 'fx4', 'fx2', 'comfort', 'value leader', 'convenience', 'easy',
                     'value package', 'journey']
        }

        for category, keywords in categories.items():
            if any(kw in t for kw in keywords):
                return category

        return 'other'

    def categorize_body(self, carrozzeria):
        categories = {
            "sedan": ["sedan", "g sedan"],
            "suv": ["suv"],
            "hatchback": ["hatchback"],
            "van": ["minivan", "van", "e-series van", "transit van", "promaster cargo van", "ram van"],
            "coupé": ["coupe", "g coupe", "genesis coupe", "koup", "cts coupe", "elantra coupe", "q60 coupe",
                      "g37 coupe", "cts-v coupe"],
            "cabriolet": ["convertible", "g convertible", "beetle convertible", "q60 convertible",
                             "g37 convertible"],
            "station wagon": ["wagon", "tsx sport wagon", "cts wagon", "cts-v wagon"],
            "pickup": ["regular cab", "regular-cab", "extended cab", "king cab", "access cab", "xtracab", "cab plus",
                       "cab plus 4", "crew cab", "supercab", "quad cab", "double cab", "crewmax cab", "mega cab", "supercrew"],
        }

        for category, values in categories.items():
            if carrozzeria in values:
                return category

        return "other"


    def preprocess(self, data, output_filename="processed_data.csv"):
        data = data.drop(columns=['vin', 'seller', 'saledate', 'state', 'mmr'], errors='ignore')
        data = data.rename(columns={
            "year": "anno produzione",
            "make": "marca",
            "model": "modello",
            "trim": "allestimento",
            "body": "carrozzeria",
            "transmission": "trasmissione",
            "condition": "condizione",
            "odometer": "chilometraggio",
            "color": "colorazione",
            "interior": "colore interni",
            "sellingprice": "prezzo"
        })

        soglia = 500
        counts = data['marca'].value_counts()
        valori_validi = counts[counts >= soglia].index
        data = data[data['marca'].isin(valori_validi)]

        data = data[data['prezzo'] > 500]

        str_cols = data.select_dtypes(include=['object', 'string']).columns
        data[str_cols] = data[str_cols].apply(lambda x: x.str.lower().str.strip())

        data['marca'] = data['marca'].replace(self.brand_mapping)

        data['condizione'] = data['condizione'].apply(
            lambda x: int(Decimal(x).to_integral_value(rounding=ROUND_HALF_DOWN)) if pd.notnull(x) else pd.NA
        )

        data['chilometraggio'] = data['chilometraggio'].apply(
            lambda x: int(Decimal(x).to_integral_value(rounding=ROUND_HALF_DOWN)) if pd.notnull(x) else pd.NA
        )

        data.loc[data['allestimento'] == data['modello'], 'allestimento'] = 'base'

        data.loc[data['colorazione'] == '—', 'colorazione'] = None
        data.loc[data['colore interni'] == '—', 'colore interni'] = None

        data['allestimento'] = data['allestimento'].apply(self.categorize_trim)
        data['carrozzeria'] = data['carrozzeria'].apply(self.categorize_body)

        output_path = os.path.join(self.output_dir, output_filename)
        data.to_csv(output_path, index=False)

        return output_path
