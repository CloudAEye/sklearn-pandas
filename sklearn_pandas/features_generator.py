def gen_features(rows, classes=None, return_dict=False, prefix='', suffix=''):
    """Generates a feature definition list which can be passed
    into DataFrameMapper
    
    Params:
    columns     a list of row names to generate features for.
    
    classes     a list of classes for each feature, a list of dictionaries with
                transformer class and init parameters, or None.
                If list of classes is provided, then each of them is
                instantiated with default arguments. Example:
                    classes = [StandardScaler, LabelBinarizer]
                If list of dictionaries is provided, then each of them should
                have a 'class' key with transformer class. All other keys are
                passed into 'class' value constructor. Example:
                    classes = [
                        {'class': StandardScaler, 'with_mean': False},
                        {'class': LabelBinarizer}
                    }]
                If None value selected, then each feature left as is.
    
    return_dict if True, returns a dictionary mapping column names to their
                feature definitions. If False (default), returns a list of
                tuples as before. Useful for feature introspection and debugging.
    
    prefix      add prefix to transformed column names
    
    suffix      add suffix to transformed column names.
    """
    if classes is None:
        feature_defs = [(column, None) for column in columns]
        if return_dict:
            return {col: None for col in columns}
        return feature_defs
    
    feature_defs = []
    feature_dict = {}
    
    for row in rows:
        feature_transformers = []
        arguments = {}
        if prefix and prefix != "":
            arguments['prefix'] = prefix
        if suffix and suffix != "":
            arguments['suffix'] = suffix
        classes_list = [cls for cls in classes if cls is not None]
        
        if not classes_list:
            feature_def = (column, None, arguments)
            feature_defs.append(feature_def)
            feature_dict[column] = feature_def
        else:
            for definition in classes_list:
                if isinstance(definition, dict):
                    params = definition.copy()
                    klass = params.pop('class')
                    feature_transformers.append(klass(**params))
                else:
                    feature_transformers.append(definition())
            
            if not feature_transformers:
                feature_transformers = None
            
            feature_def = (column, feature_transformers, arguments)
            feature_defs.append(feature_def)
            feature_dict[column] = feature_def
    
    if return_dict:
        return feature_dict
    return feature_defs
