import flask
import pickle
from urllib.request import urlopen
from urllib.parse import quote
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Fragments
import pubchempy as pcp
import pandas as pd
import numpy as np

# Use pickle to load in the pre-trained model.

with open(f'model/model_DT.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        chemical = flask.request.form['chemical']
        numericalPrediction = getPred(chemical)
        if(numericalPrediction == 2):
            prediction = "The compound is a tolerable pesticide."
        elif(numericalPrediction == 1):
            prediction = "The compound is not a tolerable pesticide."
        else:
            prediction = "Invalid compound."
        return flask.render_template('main.html',
                                     original_input={'chemical':chemical},
                                     result=prediction,
                                     )

def getPred(chemical):
    try:
        numericalPrediction = model.predict(genDesc(chemical))[0]
        return numericalPrediction
    except:
        return 3

def genDesc(name):
    sm = CIRconvert(name)
    mol = Chem.MolFromSmiles(sm)
    desc = []

    #RDKit descriptors
    desc.append(Descriptors.ExactMolWt(mol))
    desc.append(Descriptors.HeavyAtomMolWt(mol))
    desc.append(Descriptors.NumValenceElectrons(mol))
    desc.append(Lipinski.HeavyAtomCount(mol))
    desc.append(Lipinski.NHOHCount(mol))
    desc.append(Lipinski.NOCount(mol))
    desc.append(Lipinski.NumAliphaticCarbocycles(mol))
    desc.append(Lipinski.NumAliphaticHeterocycles(mol))
    desc.append(Lipinski.NumAliphaticRings(mol))
    desc.append(Lipinski.NumAromaticCarbocycles(mol))
    desc.append(Lipinski.NumAromaticHeterocycles(mol))
    desc.append(Lipinski.NumAromaticRings(mol))
    desc.append(Lipinski.NumHAcceptors(mol))
    desc.append(Lipinski.NumHDonors(mol))
    desc.append(Lipinski.NumHeteroatoms(mol))
    desc.append(Lipinski.NumRotatableBonds(mol))
    desc.append(Lipinski.NumSaturatedCarbocycles(mol))
    desc.append(Lipinski.NumSaturatedHeterocycles(mol))
    desc.append(Lipinski.NumSaturatedRings(mol))
    desc.append(Lipinski.RingCount(mol))
    desc.append(Fragments.fr_halogen(mol))
    #return desc.values

    descDF = pd.DataFrame(np.reshape(desc, (1, -1)), columns = ['ExactMolWt', 'HeavyAtomMolWt', 'NumValenceElectrons', 'HeavyAtomCount', 'NHOHCount', 
                        'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 
                        'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 
                        'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 
                        'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'Halogens'], index = [pcp.get_compounds(sm, "smiles")[0].cid])

    #PubChem descriptors
    features_2D = ["MolecularWeight", "XLogP", "ExactMass", "TPSA",
                   "Complexity", "Charge", "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount",
               "HeavyAtomCount", "IsotopeAtomCount", "AtomStereoCount", "DefinedAtomStereoCount", "UndefinedAtomStereoCount",
               "BondStereoCount", "DefinedBondStereoCount", "UndefinedBondStereoCount", "CovalentUnitCount"]
    desc_2D = pcp.get_properties(features_2D, sm, "smiles", as_dataframe=True)

    features_3D = ["Volume3D", "XStericQuadrupole3D", "YStericQuadrupole3D", "ZStericQuadrupole3D", "FeatureCount3D",
               "FeatureAcceptorCount3D", "FeatureDonorCount3D", "FeatureAnionCount3D", "FeatureCationCount3D",
               "FeatureRingCount3D", "FeatureHydrophobeCount3D", "ConformerModelRMSD3D", "EffectiveRotorCount3D",
               "ConformerCount3D"]
    desc_3D = pcp.get_properties(features_3D, sm, "smiles", as_dataframe=True)
    
    overall = pd.concat([descDF, desc_2D, desc_3D], axis=1)
    return overall

def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'Did not work'