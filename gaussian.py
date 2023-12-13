import numpy as np
EPS = 2.2204e-16
def st_get_gaussmaps(height,width,nb_gaussian):
    e = height / width
    e1 = (1 - e) / 2
    e2 = e1 + e

    mu_x = np.repeat(0.5,nb_gaussian,0)
    mu_y = np.repeat(0.5,nb_gaussian,0)

    sigma_x = e*np.array(np.arange(1,9))/16
    sigma_y = sigma_x

    x_t = np.dot(np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width)))
    y_t = np.dot(np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width)))

    x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
    y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

    gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + EPS) * \
               np.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + EPS) +
                       (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + EPS)))

    return gaussian

def dy_get_gaussmaps(height,width,nb_gaussian):
    e = height / width
    e1 = (1 - e) / 2
    e2 = e1 + e

    mu_x = np.repeat(0.5,nb_gaussian,0)
    mu_y = np.repeat(0.5,nb_gaussian,0)


    # sigma_x = np.array([1/4,1/4,1/4,1/4,
    #                     1/2,1/2,1/2,1/2])
    sigma_y = np.array([1 / 32, 1 / 32, 1 / 32, 1 / 32,
                        1 / 16, 1 / 16, 1 / 16, 1/ 16])
    sigma_x = e * np.array([1 / 32, 1 / 16, 1 / 8, 1 / 4,
                            1 / 32, 1 / 16, 1 / 8, 1 / 4])
    # sigma_y = e*np.array([1 / 16, 1 / 8, 3 / 16, 1 / 4,
    #                       1 / 8, 1 / 4, 3 / 8, 1 / 2])

    x_t = np.dot(np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width)))
    y_t = np.dot(np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width)))

    x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
    y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

    gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + EPS) * \
               np.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + EPS) +
                       (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + EPS)))

    return gaussian

def get_guasspriors(type='st', b_s=2, shape_r=60, shape_c=80, channels = 8):

    if type == 'dy':
        ims = dy_get_gaussmaps(shape_r, shape_c, channels)
    else:
        ims = st_get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims,b_s,axis=0)

    return ims

def get_guasspriors_3d(type = 'st', b_s = 2, time_dims=7,shape_r=60, shape_c=80, channels = 8):

    if type == 'dy':
        ims = dy_get_gaussmaps(shape_r, shape_c, channels)
    else:
        ims = st_get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, time_dims, axis=0)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, b_s, axis=0)
    return ims