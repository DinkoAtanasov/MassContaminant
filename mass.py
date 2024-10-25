#!/usr/bin/env python

from __future__ import unicode_literals

import re
import numpy as np
import pandas as pd

from typing import List


class Ame:
    """
    Class **Ame**
    
    Handling of Atomic Mass Evaluation database file.
    The class supports atomic mass calculations given 
    a standardized chemical element symbols.
    It supports calculation of molecule atomic mass 
    and conversion to mass excess.

    v1.0 Created  by D. Atanasov @ 16.09.2014
    v2.0 Modified by D. Atanasov @ 14.02.2017
    v3.0 Modified to have OOP by D. Atanasov @ 21.03.2017
    v4.0 Modified to python3 D. Atanasov @ 08.05.2019
    v4.1 Modified AME load to match 2020 standard D.Atanasov @ 15.10.2022

    """

    symbols = {'n': 0, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11,
               'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21,
               'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31,
               'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41,
               'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
               'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61,
               'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
               'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
               'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
               'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
               'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
               'Rg': 111, 'Cn': 112, 'Ed': 113, 'Fl': 114, 'Ef': 115, 'Lv': 116, 'Eh': 117, 'Ei': 118}

    def __init__(self):
        self.idx = []
        self.scale = 931494.061                          # [amu]->[keV]  #
        self.me = 548.57990946                           # Electron mass #
        self.me_unc = 0.00000022                         # uncertainty   #
        self.df = self.load_table()
        self.add_charge_col()

    def evaluate_expr(self, expr: str, flag=False) -> (str, int, int, int):
        """
        Evaluate a given expression of atomic number and symbol. 
        Can evaluate a complex molecule as well.

        :param expr: with a structure (40Ca:19F), (H1H1O16), 85Rb, 136Cd etc.
        :param flag: Flag to indicate if returned values are lists 
        or combined (str, int, int, int)
        :return: (combined) element symbol, (combined) multiplicity, 
                 (combined) atomic number, (combined) proton number
        """
        if ':' in expr:
            m, a, sym = self.get_molecule_info(expr)
        else:
            a, sym = self.get_number_symbol(expr)
            m = [1] * len(a)
        z = [self.symbols[isym] for isym in sym]
        if flag:
            return ''.join(sym), sum(m), sum(a), sum(z)
        else:
            return sym, m, a, z

    @staticmethod
    def get_number_symbol(item: str) -> (List, List):
        """
        Get the Atomic number and the Element symbol 
        from a specific string (does not look for multiplicity)
        Examples: 85Rb, 136Cd, (coming from molecule string) 1O16 etc.

        :param item:
        :return: Numbers, Symbols
        """
        template = '[0-9][0-9][0-9]|[0-9][0-9]|[0-9]'
        numbers = [int(i) for i in re.findall(template, item)]
        symbols = re.findall('[A-Z][a-z]|[A-Z]', item)
        return numbers, symbols

    def get_molecule_info(self, expr: str) -> (List, List, List):
        """
        Get the Molecule Info from MM8 string with multiplicity
        Examples: (2H1:1O16), (Sr82:F19), 
        if the multiplicity is missing it's assumed 1.
        
        :param expr:
        :return: Multiplicities, Atomic numbers, Element Symbols
        """
        multi = []
        sym = []
        a = []
        if ':' not in expr:
            # Something is wrong with the item string, return empty results
            return multi, a, sym

        tmp_nbrs, sym = self.get_number_symbol(expr)
        if len(tmp_nbrs) > len(sym):
            multi = [tmp_nbrs[i] for i in range(0, len(tmp_nbrs), 2)]
            a = [tmp_nbrs[i] for i in range(1, len(tmp_nbrs)+1, 2)]
        else:
            multi = [1]*len(tmp_nbrs)
            a = tmp_nbrs
        return multi, a, sym

    @staticmethod
    def load_table(ame='mass_1.mas20.txt') -> pd.DataFrame:
        """
        Load Atomic Mass Evaluation table and returns Pandas DataFrame.

        .. note::

            If a new table is uploaded, please ensure
            the column widths are correct
            and compare the input/output is being correctly read.

        :param ame: 'mas20.txt' As provided by the AME group.
        :return: Pandas DataFrame
        """
        col_width = [1, 3, 5, 5, 5, 4, 4, 17, 13, 13, 10, 3, 12, 10, 4, 14, 11]
        ns = ['cc', 'NZ', 'N', 'Z', 'A', 'EL', 'o', 'ME', 'MEunc', 'BE / keV', 
              'BEunc / keV', 'B', 'BDE / keV', 'BDE unc / keV', 'am(int)', 
              'am(rest)', 'amunc']
        df = pd.read_fwf(ame, widths=col_width, skiprows=36, names=ns)
        df['ame_tot'] = df['am(int)']*1E6 + df['am(rest)']

        df = df.drop(columns=['cc', 'NZ', 'o', 'BE / keV', 
                              'BEunc / keV',
                              'B', 'BDE / keV', 
                              'BDE unc / keV', 'am(int)', 'am(rest)'])
        return df

    def add_charge_col(self) -> None:
        nrows = len(self.df)
        ncols = len(self.df.columns)
        self.df.insert(ncols, 'charge', nrows * [1])

    def add_entry_table(self, expr: str) -> None:
        """
        Add new entry to AME table. 
        It is used to create a molecule or missing information in the table.

        :param expr: Expression of the atom or molecule
        :return: None
        """
        el, amass, amunc, a, z, n, q = self.get_ame_mass(expr)
        mexcess, meunc = self.mass_excess(amass, amunc, a)
        cols = [n, z, a, expr, mexcess, meunc, amass, amunc, q]
        ns = ['N', 'Z', 'A', 'EL', 'ME', 'MEunc', 'ame_tot', 'amunc', 'charge']
        new_entry = pd.DataFrame([cols], columns=ns)
        self.df = pd.concat([self.df, new_entry], sort=False, ignore_index=True)

    def get_ame_mass(self, el_expr: str) -> (str, float, float, float, float, float, float): 
        """
        Checks AME Table for existing of the element 
        with the given atomic number. Calculates the mass. 
        If a list is provided to the function (such as a molecule) it calculates
        the summed mass of the constituents.

        :param el_expr:
        :return: el_expr, ame_mass, np.sqrt(ame_unc), a, z, n, q
        """
        self.idx = []
        ame_mass = 0.0
        ame_unc = 0.0
        q = 1

        label, multiplicity, atomic_number, z = self.evaluate_expr(el_expr)

        for i in range(len(label)):
            cond1 = self.df['EL'] == label[i] 
            cond2 = self.df['A'] == int(atomic_number[i])
            self.idx.append(self.df[cond1 & cond2].index.tolist())

        if not bool(self.idx):
            print("Element or Atomic number not found in database")
        else:
            for i in range(len(self.idx)):
                mul = float(multiplicity[i])
                part = self.df.at[self.idx[i][0], 'ame_tot']
                unc_part = self.df.at[self.idx[i][0], 'amunc']
                ame_mass += mul * float(part)
                ame_unc += mul * float(unc_part)**2
        sym = ''.join(label)
        a = sum([ai*mi for ai, mi in zip(atomic_number, multiplicity)])
        z = sum(z)
        n = a - z
        return sym, ame_mass, np.sqrt(ame_unc), a, z, n, q

    def get_extra_mass(self, expression: str) -> float:
        """
        Checks AME Table for existing of the element with the given 
        atomic number. Calculates the mass. 
        If a list is provided to the function (such as a molecule) 
        it calculates the summed mass of the constituents.

        :param expression:
        :return: atomic_mass
        """
        _, ame_mass, *_ = self.get_ame_mass(expression)
        return ame_mass

    def get_ion_mass(self, el_symbol: str, charge=1) -> (float, float):
        """
        Lookup in AME Table the atomic mass and return the Ion mass

        :param el_symbol:
        :param charge:
        :return:
        """
        el_expr, mass_atom, mass_atom_unc, *_ = self.get_ame_mass(el_symbol)
        ion_mass = (mass_atom - charge*self.me)
        unc_ion_mass = np.sqrt((mass_atom_unc**2) + (charge * self.me_unc)**2)
        return ion_mass, unc_ion_mass

    def mass_excess(self, atomic_mass, unc_atomic_mass, mass_number) -> (float, float):
        """
        Calculate the Mass excess from a given atomic mass and atomic mass number.

        :param atomic_mass:
        :param unc_atomic_mass:
        :param mass_number:
        :return: mass_excess, unc_mass_excess
        """
        mass_excess = (atomic_mass * 1E-6 - mass_number) * self.scale
        mass_excess_unc = (unc_atomic_mass * 1E-6) * self.scale
        return mass_excess, mass_excess_unc

    def update_mass_excess(self, eframe: pd.DataFrame, aeff: int) -> None:
        """
        Update the Mass excess from external DataFrame

        :param eframe: external DataFrame
        :param aeff: effective mass number
        """
        eframe['ME'] = (eframe['ame_tot'] * 1E-6 - aeff) * self.scale
        eframe['MEunc'] = (eframe['amunc'] * 1E-6) * self.scale

        eframe['ME'] = round(eframe['ME'], 3)
        eframe['MEunc'] = round(eframe['MEunc'], 3)

    def calc_isomer_mass(self, el_id: str, ex_erg: float) -> pd.DataFrame:
        q = 1
        sym, m, a, z = self.evaluate_expr(el_id, flag=True)
        n = a - z
        cond = (self.df['EL'] == sym) & (self.df['A'] == int(a))
        i = self.df[cond].index.tolist()[0]

        # self.add_entry_table(el_id+'m')
        me = float(self.df.at[i, 'ME']) + ex_erg
        amass = ((me / self.scale) + a) * 1E6
        # fake uncertainties...
        meunc = 0.
        amunc = 0.
        cols = [n, z, a, sym+'^m', me, meunc, amass, amunc, q]
        ns = ['N', 'Z', 'A', 'EL', 'ME', 'MEunc', 'ame_tot', 'amunc', 'charge']
        return pd.DataFrame([cols], columns=ns)

    def get_isobars(self, atomic_number):
        """
        Checks AME Table for existing of the element 
        with the given atomic number. Calculates the mass. 
        If a list is provided to the function (such as a molecule) it calculates 
        the summed mass of the constituents.

        :param atomic_number:
        :return: Pandas DataFrame
        """

        idx = self.df[self.df['A'] == atomic_number].index.tolist()

        isobars = np.empty((len(idx), 3), dtype=object)
        if len(idx) == 0:
            print("No isobars found! Please check the atomic Number")
        else:
            for i in range(len(idx)):
                isobars[i][0] = self.df.at[idx[i], 'EL']
                isobars[i][1] = float(self.df.at[idx[i], 'ame_tot'])
                isobars[i][2] = float(self.df.at[idx[i], 'amunc'])
        return isobars

    def get_a_info(self, atomic_number):
        """
        Checks AME Table for existing of the element with 
        the given atomic number. Calculates the mass. 
        If a list is provided to the function (such as a molecule) it calculates
        the summed mass of the constituents.

        :param atomic_number:
        :return:
        """
        return self.df[self.df['A'] == atomic_number].to_html()

    def get_z_info(self, z_number):
        """
        Checks AME Table for existing of the element 
        with the given atomic number. Calculates the mass. 
        If a list is provided to the function (such as a molecule) it calculates
        the summed mass of the constituents.

        :param z_number:
        :return:
        """
        return self.df[self.df['Z'] == z_number].to_html()

    def get_n_info(self, n_number):
        """
        Checks AME Table for existing of the element 
        with the given atomic number. Calculates the mass. 
        If a list is provided to the function (such as a molecule) it calculates
        the summed mass of the constituents.

        :param n_number:
        :return:
        """
        return self.df[self.df['N'] == n_number].to_html()

    def get_el_info(self, el):
        """
        Checks AME Table for existing of the element 
        with the given atomic number. Calculates the mass. 
        If a list is provided to the function (such as a molecule) it calculates
        the summed mass of the constituents.

        :param el:
        :return:
        """
        return self.df[self.df['EL'] == el].to_html()
