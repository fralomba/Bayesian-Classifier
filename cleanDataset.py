import numpy as np

def encodeRealColumns(columns_to_be_encoded):

    max_real_columns = np.amax(columns_to_be_encoded, axis=0)
    min_real_columns = np.amin(columns_to_be_encoded, axis=0)

    deltas = [max_real_columns[i] - min_real_columns[i] for i in range(len(max_real_columns))]
    steps = [delta/4 for delta in deltas]

    oneHotEncoded1on4s = []
    oneHotEncoded2on4s = []
    oneHotEncoded3on4s = []
    oneHotEncoded4on4s = []


    for i in range(columns_to_be_encoded.shape[1]):
        oneHotEncoded1on4s.append(np.logical_and(columns_to_be_encoded[:, i] >= min_real_columns[i], columns_to_be_encoded[:, i] < min_real_columns[i] + steps[i]))
        oneHotEncoded2on4s.append(np.logical_and(columns_to_be_encoded[:, i] >= min_real_columns[i] + steps[i], columns_to_be_encoded[:, i] < min_real_columns[i] + 2*steps[i]))
        oneHotEncoded3on4s.append(np.logical_and(columns_to_be_encoded[:, i] >= min_real_columns[i] + 2*steps[i], columns_to_be_encoded[:, i] < min_real_columns[i] + 3*steps[i]))
        oneHotEncoded4on4s.append(np.logical_and(columns_to_be_encoded[:, i] >= min_real_columns[i] + 3*steps[i], columns_to_be_encoded[:, i] < min_real_columns[i] + 4*steps[i]))

    OneHotEncoding = np.asarray([oneHotEncoded1on4s,oneHotEncoded2on4s,oneHotEncoded3on4s,oneHotEncoded4on4s]).astype(int)
    return OneHotEncoding



