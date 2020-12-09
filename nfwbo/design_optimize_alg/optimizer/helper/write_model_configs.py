def write_model_configs(xbounds, zbounds, oriparams):
    name = oriparams['warping_manner']
    params = dict()

    if name == 'boca':
        dimension_components = []
        xdim = xbounds.shape[1]
        zdim = zbounds.shape[1]

        X_indices = dict() # capital X refers to xz
        for i in range(xdim):
            dimension_components.append('x' + str(i))
            X_indices['x' + str(i)] = [i]
        for i in range(zdim):
            dimension_components.append('z' + str(i))
            X_indices['z' + str(i)] = [i + xdim]

        warping_indices = dict()
        for k , v in X_indices.items():
            if k.startswith('x'):
                warping_indices[k] = [] # using **empty** list when no warping
            elif k.startswith('z'):
                warping_indices[k] = [] #v

        warping_hid_dims = dict()
        for k, v in warping_indices.items():
            if len(v) > 0:
                warping_hid_dims[k] = oriparams['hiddens']
            else:
                warping_hid_dims[k] = []

        warping_out_dim = dict()
        for k, v in warping_indices.items():
            if len(v) > 0:
                warping_out_dim[k] = oriparams['out_dim']
            else:
                warping_out_dim[k] = None

        X_warped_indices = dict()
        last_indices = 0
        for dc in dimension_components:
            if warping_out_dim[dc] is not None:
                X_warped_indices[dc] = list(range(last_indices, last_indices + warping_out_dim[dc]))
                last_indices += warping_out_dim[dc]
            else:
                last_indices += len(X_indices[dc])

    elif name == 'fidelitywarping':
        dimension_components = []
        xdim = xbounds.shape[1]
        zdim = zbounds.shape[1]

        X_indices = dict() # capital X refers to xz
        for i in range(xdim):
            dimension_components.append('x' + str(i))
            X_indices['x' + str(i)] = [i]
        for i in range(zdim):
            dimension_components.append('z' + str(i))
            X_indices['z' + str(i)] = [i + xdim]

        warping_indices = dict()
        for k , v in X_indices.items():
            if k.startswith('x'):
                warping_indices[k] = [] # using **empty** list when no warping
            elif k.startswith('z'):
                warping_indices[k] = v #v

        warping_hid_dims = dict()
        for k, v in warping_indices.items():
            if len(v) > 0:
                warping_hid_dims[k] = oriparams['hiddens']
            else:
                warping_hid_dims[k] = []

        warping_out_dim = dict()
        for k, v in warping_indices.items():
            if len(v) > 0:
                warping_out_dim[k] = oriparams['out_dim'] #None
            else:
                warping_out_dim[k] = None

        X_warped_indices = dict()
        last_indices = 0
        for dc in dimension_components:
            if warping_out_dim[dc] is not None:
                X_warped_indices[dc] = list(range(last_indices, last_indices + warping_out_dim[dc]))
                last_indices += warping_out_dim[dc]
            else:
                last_indices += len(X_indices[dc])
    elif name == 'fabolas':
        params['k_dim'] = xbounds.shape[1] + zbounds.shape[1]
        return params
    else:
        raise NotImplementedError

    params['X_indices'] = X_indices
    params['warping_indices'] = warping_indices
    params['warping_hid_dims'] = warping_hid_dims
    params['warping_out_dim'] = warping_out_dim
    params['X_warped_indices'] = X_warped_indices
    w_dim = 0
    for k, v in warping_out_dim.items():
        if v is not None:
            w_dim += v
        else:
            w_dim += len(X_indices[k])
    params['w_dim'] = w_dim
    params['k_dim'] = w_dim
    return params