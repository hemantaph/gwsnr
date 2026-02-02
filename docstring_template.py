# -*- coding: utf-8 -*-
"""
DOCSTRING TEMPLATE FILE - Generated from gwsnr/core/gwsnr.py patterns.

This file provides standardized docstring templates for documenting Python code
in the gwsnr package. It follows NumPy/SciPy docstring conventions and includes
RST (reStructuredText) formatting compatible with Sphinx documentation.

Usage:
    Copy the relevant template sections for your use case:
    - Module docstrings
    - Class docstrings (with parameter tables, method tables, attribute tables)
    - Method/function docstrings  
    - Property docstrings

Copyright (C) 2025 Your Name. Distributed under MIT License.

AI prompt to use this template: Rewrite docstrings (including inline docstrings) of *.py using docstring_template.py. Create properties of the Instance Attributes if it is not there. Convert some of the Instance Methods to private methods with underscore prefix and remove them from the class docstring. If there are any code inconsistencies or docstring inconsistencies, please point them out without making any changes.
"""

import numpy as np


# -------------------------------------------====
# CONSTANTS
# -------------------------------------------====
# Constants should be defined at module level with brief inline comments
SPEED_OF_LIGHT = 299792458.0  # m/s
GRAVITATIONAL_CONSTANT = 6.67408e-11  # m^3 kg^-1 s^-2
PI = np.pi


# -------------------------------------------====
# CLASS TEMPLATE
# -------------------------------------------====
class TemplateClass:
    """
    One-line summary describing the class purpose.

    Extended description providing more details about the class functionality,
    its purpose, and how it fits within the larger system. Can span multiple
    paragraphs if needed.

    Key Features:
    - Feature 1 description \n
    - Feature 2 description \n
    - Feature 3 description \n

    Parameters
    ----------
    param1 : ``int``
        Description of parameter 1. \n
        default: 10
    param2 : ``float``
        Description of parameter 2. \n
        default: 1.0
    param3 : ``str``
        Description of parameter 3. Options: \n
        - 'option_a': Description of option A \n
        - 'option_b': Description of option B \n
        - 'option_c': Description of option C \n
        default: 'option_a'
    param4 : ``dict``
        Complex parameter with structure description. \n
        Options: \n
        - None: Description of None behavior \n
        - {'key1': 'value1'}: Custom configuration \n
    param5 : ``list`` or ``None``
        Custom list of objects. Optional. \n
        Options: \n
        - None: Use defaults \n
        - ['item1', 'item2']: Custom list
    param6 : ``bool``
        Boolean flag description. \n
        default: True
    param7 : ``numpy.ndarray``
        Array parameter description. Optional.

    Examples
    --------
    Basic usage:
    
    >>> from module import TemplateClass
    >>> obj = TemplateClass()
    >>> result = obj.method(arg1=10, arg2=20)
    >>> print(f"Result: {result}")

    Advanced usage with custom configuration:
    
    >>> # Custom configuration example
    >>> import module
    >>> from module import TemplateClass
    >>> custom_obj = module.CustomClass(
            name='CustomName',
            value=42.0)
    >>> obj = TemplateClass(param4=dict(key='value'), param5=[custom_obj])


    Instance Methods
    ----------------
    TemplateClass class has the following methods:  \n
    +------------------------------------------------+------------------------------------------------+
    | Method                                         | Description                                    |
    +================================================+================================================+
    | :meth:`~method_name_1`                         | Brief description of method 1                  |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~method_name_2`                         | Brief description of method 2. Continues on    |
    |                                                | next line if needed                            |
    +------------------------------------------------+------------------------------------------------+
    | :meth:`~method_name_3`                         | Brief description of method 3                  |
    +------------------------------------------------+------------------------------------------------+

    Instance Attributes
    -------------------
    TemplateClass class has the following attributes:  \n
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | Attribute                                      | Type             | Unit  | Description                                    |
    +================================================+==================+=======+================================================+
    | :meth:`~attr1`                                 | ``int``          |       | Integer attribute description                  |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~attr2`                                 | ``float``        | Hz    | Float attribute with units                     |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~attr3`                                 | ``str``          |       | String attribute description                   |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~attr4`                                 | ``ndarray``      | M☉    | NumPy array with units                         |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~attr5`                                 | ``dict``         |       | Dictionary attribute description               |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~attr6`                                 | ``list``         |       | List attribute description                     |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~attr7`                                 | ``bool``         |       | Boolean attribute description                  |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~attr8`                                 | ``float/None``   | s     | Optional float with units                      |
    +------------------------------------------------+------------------+-------+------------------------------------------------+
    | :meth:`~attr9`                                 | ``function``     |       | Callable/function attribute                    |
    +------------------------------------------------+------------------+-------+------------------------------------------------+

    Notes
    -----
    - Important implementation note 1 \n
    - Important implementation note 2 \n
    - Important implementation note 3: more detailed explanation \n
    """

    def __init__(
        self,
        # Group 1: General settings
        param1=10,
        param2=1.0,
        param3="option_a",
        param4=None,
        # Group 2: Configuration
        param5=None,
        param6=True,
        param7=None,
    ):
        # Initialize instance attributes
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4 if param4 is not None else {}
        self.param5 = param5
        self.param6 = param6
        self.param7 = param7

    # -------------------------------------------
    # PUBLIC METHOD TEMPLATE - Simple
    # -------------------------------------------
    def simple_method(self, arg1, arg2=10.0):
        """
        One-line summary of what this method does.

        Extended description providing more context about the method's
        functionality, algorithm, or approach.

        Parameters
        ----------
        arg1 : ``float``
            Description of arg1.
        arg2 : ``float``
            Description of arg2 with default value.\n
            default: 10.0

        Returns
        -------
        ``float``
            Description of the return value.

        Notes
        -----
        Additional implementation notes if needed.
        """
        return arg1 + arg2

    # -------------------------------------------
    # PUBLIC METHOD TEMPLATE - Complex with multiple returns
    # -------------------------------------------
    def complex_method(
        self,
        param1=np.array([10.0,]),
        param2=np.array([10.0,]),
        param3=100.0,
        param4=0.0,
        param5=0.0,
        param_dict=False,
        output_file=False,
    ):
        """
        One-line summary of the complex method.

        This is the extended description explaining the purpose, functionality,
        and context of this method. It can include details about the algorithm,
        computation approach, and how it relates to other methods.

        Parameters
        ----------
        param1 : ``numpy.ndarray`` or ``float``
            Primary parameter description.\n
            default: np.array([10.0,])
        param2 : ``numpy.ndarray`` or ``float``
            Secondary parameter description.\n
            default: np.array([10.0,])
        param3 : ``numpy.ndarray`` or ``float``
            Third parameter description with units (e.g., in Mpc).\n
            default: 100.0
        param4 : ``numpy.ndarray`` or ``float``
            Fourth parameter description (e.g., angle in radians).\n
            default: 0.0
        param5 : ``numpy.ndarray`` or ``float``
            Fifth parameter description (e.g., angle in radians).\n
            default: 0.0
        param_dict : ``dict`` or ``bool``
            Parameter dictionary. If provided, overrides individual arguments.\n
            default: False\n
            Example:\n
            param_dict = {'param1': [20, 30], 'param2': [20, 25], 'param3': [100, 200]}
        output_file : ``str`` or ``bool``
            Save results to file. If True, saves with default name.\n
            default: False

        Returns
        -------
        ``dict``
            Result dictionary with keys 'result_1', 'result_2', 'result_net'.
            Values are arrays matching input size.

        Notes
        -----
        - For method variant A, parameter X is converted using formula Y
        - Total value must be within [min, max] for valid results
        - Hybrid computation uses higher-order approximations near threshold

        Examples
        --------
        >>> obj = TemplateClass()
        >>> result = obj.complex_method(param1=30.0, param2=25.0, param3=100.0)
        >>> print(f"Network result: {result['result_net'][0]:.2f}")
        
        >>> # Multiple items with parameter dictionary
        >>> params = {'param1': [20, 30], 'param2': [20, 25], 'param3': [100, 200]}
        >>> result = obj.complex_method(param_dict=params)
        """
        if not param_dict:
            param_dict = {
                "param1": param1,
                "param2": param2,
                "param3": param3,
                "param4": param4,
                "param5": param5,
            }

        # Method implementation
        result = {
            "result_1": np.array(param_dict["param1"]) + np.array(param_dict["param2"]),
            "result_2": np.array(param_dict["param3"]),
            "result_net": np.zeros_like(np.array(param_dict["param1"])),
        }

        return result

    # -------------------------------------------
    # PRIVATE METHOD TEMPLATE
    # -------------------------------------------
    def _private_helper_method(self, data, config):
        """
        Brief description of private method functionality.

        Extended description of what this internal helper method does
        and how it's used within the class.

        Parameters
        ----------
        data : ``numpy.ndarray``
            Input data array to process.
        config : ``dict``
            Configuration dictionary with keys: 'setting1', 'setting2'.

        Returns
        -------
        processed_data : ``numpy.ndarray``
            Processed output data array.

        Notes
        -----
        - Uses :func:`external_function` for computation \n
        - Called by :meth:`~main_method` for processing \n
        - Modifies :attr:`~internal_state` as side effect \n
        """
        # Implementation
        processed_data = data * config.get('setting1', 1.0)
        return processed_data

    # -------------------------------------------
    # PRIVATE METHOD - Error/Placeholder
    # -------------------------------------------
    def _placeholder_method(self, **kwargs):
        """
        Print error message when method is called without required setup.

        This placeholder method is assigned when certain functionality
        is not available based on initialization parameters.

        Parameters
        ----------
        **kwargs : ``dict``
            Arbitrary keyword arguments (ignored).

        Raises
        ------
        ``ValueError``
            Always raised, suggesting proper initialization.
        """
        raise ValueError(
            'Feature not available. Please initialize with correct parameters.'
        )

    # -------------------------------------------
    # PRIVATE METHOD - Initialization Helper
    # -------------------------------------------
    def _initialization_helper(self, path, create_new, data_list):
        """
        Set up resources for the class functionality.

        This method manages the creation and loading of required resources.
        It checks for existing files, generates missing ones, and loads data
        for runtime use.

        Parameters
        ----------
        path : ``str``
            Directory path for storing resource files.
        create_new : ``bool``  
            If True, generates new resources regardless of existing files.
        data_list : ``list``
            List of data objects for processing.

        Returns
        -------
        resource_paths : list
            File paths to resource files for all items.

        Notes
        -----
        - Uses :func:`resource_check` to identify missing resources \n
        - Calls :meth:`_generate_resource` to create new data \n
        - Loads data into :attr:`~resource_list` for runtime use \n
        """
        # Implementation
        return []

    # -------------------------------------------
    # PRIVATE METHOD - Complex Initialization
    # -------------------------------------------
    def _complex_initialization(self, config_dict, item_list, frequency, model_name):
        """
        Initialize complex models and scalers for computation.

        Loads pre-trained models, scalers, and correction parameters
        for each item. Validates that model parameters match current configuration.

        Parameters
        ----------
        config_dict : ``dict`` or ``str`` or ``None``
            Dictionary or file path containing model paths for each item.
            If None, uses default models from package resources.
            Expected structure: {item_name: {'model_path': str, 'scaler_path': str, 
            'frequency': float, 'model_name': str}}.
        item_list : ``list`` of ``str``
            Item names requiring models (e.g., ['A', 'B', 'C']).
        frequency : ``float``
            Frequency in Hz. Must match model training configuration.
        model_name : ``str``
            Model name. Must match model training configuration.

        Returns
        -------
        model_dict : ``dict``
            Loaded models {item_name: model}.
        scaler_dict : ``dict``
            Feature preprocessing scalers {item_name: scaler}.
        correction_params : ``dict``
            Post-prediction correction parameters {item_name: {'slope': float, 'intercept': float}}.
        catalogue : dict
            Complete model configuration and paths for all items.

        Raises
        ------
        ValueError
            If model not available for item, or if model parameters don't match 
            current configuration.

        Notes
        -----
        - Loads models from package resources if file paths don't exist locally \n
        - Validates parameter compatibility before loading \n
        - Correction parameters improve prediction accuracy via linear correction \n
        """
        # Implementation
        return {}, {}, {}, {}

    # -------------------------------------------
    # PRIVATE METHOD - Verbose Printing
    # -------------------------------------------
    def _print_configuration(self, verbose=True):
        """
        Print all parameters and configuration of the class instance.

        Displays computational settings, model configuration, data setup,
        and parameter ranges for verification and debugging.

        Parameters
        ----------
        verbose : ``bool``
            If True, print all parameters to stdout. If False, suppress output.\n
            default: True

        Notes
        -----
        Printed information includes: \n
        - Computational: processors, method\n
        - Model: approximant, frequencies, sampling rate\n
        - Data: names and configurations\n
        - Ranges: bounds with cutoffs\n
        - Resolution: grid resolutions and bounds (when applicable)

        Called automatically during initialization when verbose=True.
        """
        if verbose:
            print(" \nConfiguration parameters: \n")
            print(f"param1: {self.param1}")
            print(f"param2: {self.param2}")

    # -------------------------------------------
    # STATIC/CLASS METHOD TEMPLATE
    # -------------------------------------------
    @staticmethod
    def calculate_threshold(max_value, min_frequency):
        """
        Calculate maximum threshold based on minimum frequency.

        This method finds the maximum value where computation remains valid
        at the given minimum frequency. A safety factor is applied.

        Parameters
        ----------
        max_value : ``float``
            User-specified maximum value.
        min_frequency : ``float``
            Minimum frequency in Hz for computation.

        Returns
        -------
        ``float``
            Adjusted maximum value (≤ input max_value) ensuring valid computation.

        Notes
        -----
        Uses conservative estimate for threshold calculation.
        """
        # Implementation
        return min(max_value, 1.0 / min_frequency)

    # -------------------------------------------
    # PROPERTY TEMPLATE - Simple
    # -------------------------------------------
    @property
    def param1(self):
        """
        Brief description of the property.

        Returns
        -------
        param1 : ``int``
            Description of what this property represents. \n
            default: 10
        """
        return self._param1

    @param1.setter
    def param1(self, value):
        self._param1 = value

    # -------------------------------------------
    # PROPERTY TEMPLATE - With Units
    # -------------------------------------------
    @property
    def param2(self):
        """
        Property representing a physical quantity.

        Returns
        -------
        param2 : ``float``
            Description (Hz for frequency, Mpc for distance, etc.). \n
            default: 1.0
        """
        return self._param2

    @param2.setter
    def param2(self, value):
        self._param2 = value

    # -------------------------------------------
    # PROPERTY TEMPLATE - String enum
    # -------------------------------------------
    @property
    def param3(self):
        """
        Property with enumerated string options.

        Returns
        -------
        param3 : ``str``
            Description. Options: 'option_a', 'option_b', 'option_c'. \n
            default: 'option_a'
        """
        return self._param3

    @param3.setter
    def param3(self, value):
        self._param3 = value

    # -------------------------------------------
    # PROPERTY TEMPLATE - Dictionary
    # -------------------------------------------
    @property
    def param4(self):
        """
        Configuration dictionary property.

        Returns
        -------
        param4 : ``dict``
            Configuration dictionary with relevant keys and values.
        """
        return self._param4

    @param4.setter
    def param4(self, value):
        self._param4 = value

    # -------------------------------------------
    # PROPERTY TEMPLATE - List
    # -------------------------------------------
    @property
    def param5(self):
        """
        List of objects property.

        Returns
        -------
        param5 : ``list``
            List of configured objects.
        """
        return self._param5

    @param5.setter
    def param5(self, value):
        self._param5 = value

    # -------------------------------------------
    # PROPERTY TEMPLATE - Boolean
    # -------------------------------------------
    @property
    def param6(self):
        """
        Boolean flag property.

        Returns
        -------
        param6 : ``bool``
            Enable/disable feature description. \n
            default: True
        """
        return self._param6

    @param6.setter
    def param6(self, value):
        self._param6 = value

    # -------------------------------------------
    # PROPERTY TEMPLATE - NumPy Array
    # -------------------------------------------
    @property
    def param7(self):
        """
        Array data property.

        Returns
        -------
        param7 : ``numpy.ndarray``
            Array of values for computation.
        """
        return self._param7

    @param7.setter
    def param7(self, value):
        self._param7 = value

    # -------------------------------------------
    # PROPERTY TEMPLATE - Optional/Nullable
    # -------------------------------------------
    @property
    def optional_param(self):
        """
        Optional parameter that may be None.

        Returns
        -------
        optional_param : ``float`` or ``None``
            Value if set, None otherwise. Auto-computed if None. \n
            default: None
        """
        return getattr(self, '_optional_param', None)

    @optional_param.setter
    def optional_param(self, value):
        self._optional_param = value

    # -------------------------------------------
    # PROPERTY TEMPLATE - Function/Callable
    # -------------------------------------------
    @property
    def compute_function(self):
        """
        Computation function reference.

        Returns
        -------
        compute_function : ``function``
            Backend-specific computation function.
        """
        return getattr(self, '_compute_function', None)

    @compute_function.setter
    def compute_function(self, value):
        self._compute_function = value


# -------------------------------------------====
# STANDALONE FUNCTION TEMPLATE
# -------------------------------------------====
def standalone_function(
    input_array,
    config_dict=None,
    output_path=None,
    verbose=True,
):
    """
    One-line summary of what this function does.

    Extended description providing context about the function's purpose,
    algorithm, and typical use cases. Can reference related functions
    using :func:`related_function` syntax.

    Parameters
    ----------
    input_array : ``numpy.ndarray``
        Input data array to process.
    config_dict : ``dict``
        Configuration dictionary with optional settings. Optional.\n
        Default is None, which uses default configuration.
    output_path : ``str``
        Path to save output. Optional.
    verbose : ``bool``
        Print progress information. Optional.

    Returns
    -------
    result : ``numpy.ndarray``
        Processed output array.
    metadata : ``dict``
        Dictionary containing processing metadata.

    Raises
    ------
    ValueError
        If input_array is empty or config_dict has invalid keys.
    IOError
        If output_path is not writable.

    See Also
    --------
    related_function : Description of related function. \n
    TemplateClass.method : Description of related method. \n

    Notes
    -----
    - Implementation detail 1 \n
    - Implementation detail 2 \n
    - Performance consideration \n

    Examples
    --------
    Basic usage:
    
    >>> import numpy as np
    >>> from module import standalone_function
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> result, metadata = standalone_function(data)
    >>> print(result)
    [2.0, 4.0, 6.0]

    With custom configuration:
    
    >>> config = {'scale': 3.0}
    >>> result, metadata = standalone_function(data, config_dict=config)
    """
    if config_dict is None:
        config_dict = {'scale': 2.0}

    result = input_array * config_dict.get('scale', 1.0)
    metadata = {'processed': True, 'scale': config_dict.get('scale', 1.0)}

    if output_path:
        # Save to file
        pass

    if verbose:
        print(f"Processed {len(input_array)} elements")

    return result, metadata


# -------------------------------------------====
# HELPER FUNCTION TEMPLATE
# -------------------------------------------====
def _internal_helper(data, factor=1.0):
    """
    Brief description of internal helper function.

    Parameters
    ----------
    data : ``numpy.ndarray``
        Input data.
    factor : ``float``
        Scaling factor. Optional.\n
        Default: 1.0.

    Returns
    -------
    ``numpy.ndarray``
        Scaled data.
    """
    return data * factor
