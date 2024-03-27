''' Utility function for the parameters '''
import yaml
import shutil
import logging

import numpy as np

def update_keys(pars_dict: dict, newpars: dict) -> dict:
    ''' Update the values of the dictionary, if present the newpars '''
    for par_key in pars_dict.keys():
        input_val = newpars.pop(par_key, None)

        if input_val is not None:
            logging.info('Setting %s to %s', par_key, input_val)
            pars_dict[par_key] = input_val

    return pars_dict

class SnnPars():
    ''' Template for parameters '''

    def __init__(
        self,
        pars_path     : str,
        parsname      : str,
        new_pars      : dict = None,
        keys_to_update: list[str] = None,
        **kwargs
    ):

        if keys_to_update is None:
            keys_to_update = []

        # Load from YAML file
        pars_type = kwargs.pop('pars_type', 'parameters')
        setattr(self, f'_{self.__class__.__name__}__file_name', f'{parsname}.yaml')
        setattr(self, f'_{self.__class__.__name__}__file_path', f'{pars_path}/{parsname}.yaml')

        logging.info(f"Loading {pars_type} from {self.file_path}")
        with open(f'{self.file_path}') as infile:
            self.pars = yaml.safe_load(infile)

        # Update values
        if new_pars is None:
            new_pars = {}

        new_pars.update(kwargs)
        self.params_to_update = new_pars
        self.pars = update_keys(self.pars, self.params_to_update)

        for key in keys_to_update:
            update_keys(self.pars[key], self.params_to_update)

    def consistency_checks(self):
        ''' Check that all parameters listed in the file have been processes '''
        assert self.pars == {}, f'Not all parameters were assigned: {self.pars}'
        del self.pars

    def save_yaml_files(self, destination_path):
        ''' Saves the yaml files to the destination path'''
        file_src = self.file_path
        file_dst = f'{destination_path}/{self.file_name}'
        logging.info('Copying %s file to %s', file_src,  file_dst)
        shutil.copyfile(file_src, file_dst)

    # PROPERTIES
    # READ-ONLY
    def read_only_attr(attr):
        """ Read-Only """
        @property
        def prop(self):
            return getattr(self, f'_{self.__class__.__name__}__{attr}')
        @prop.setter
        def prop(self, value):
            raise ValueError('Variable %s is read-only', attr)
        return prop

    # READ-WRITE
    def read_write_attr(attr, fun = None):
        """ Read-Write """
        @property
        def prop(self):
            return getattr(self, f'_{self.__class__.__name__}__{attr}')
        @prop.setter
        def prop(self, value):
            if isinstance(value, list):
                value = np.array(value)
            logging.info('Update %s to %s', attr, value)
            setattr(self, f'_{self.__class__.__name__}__{attr}', value)

            # Optionally call additional method
            if fun is not None:
                fun(self, attr, value)

        return prop

    # Files
    file_name  : str   = read_only_attr('file_name')
    file_path  : str   = read_only_attr('file_path')