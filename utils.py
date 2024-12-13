import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def validate_smiles(smiles):
    """Validate SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def clean_data(split):
    """Clean dataset splits by removing duplicates, handling missing values, and validating SMILES."""
    cleaned_split = {}

    for key, df in split.items():
        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values with median for numeric columns and 'Unknown' for others
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')

        # Clean string data and convert binary column
        df['Drug'] = df['Drug'].astype(str)
        df['Drug'] = df['Drug'].str.strip()
        df['Y'] = df['Y'].astype(int)

        # Validate SMILES strings
        df['Valid_SMILES'] = df['Drug'].apply(validate_smiles)
        df = df[df['Valid_SMILES']]
        df = df.drop(columns=['Valid_SMILES'])

        cleaned_split[key] = df

    return cleaned_split

def check_balance(cleaned_split):
    """Check class balance in dataset splits."""
    balance_info = {}
    for key, df in cleaned_split.items():
        y_distribution = df['Y'].value_counts(normalize=True)
        is_balanced = not ((y_distribution.min() < 0.4) or (y_distribution.max() > 0.6))
        balance_info[key] = {
            'distribution': y_distribution,
            'is_balanced': is_balanced
        }
    return balance_info

def generate_fingerprints(smiles):
    """Generate Morgan fingerprints for SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Radius of 2 to check interactions 2 bonds away, 2048 bits
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return list(fp)
    else:
        return [0] * 2048
