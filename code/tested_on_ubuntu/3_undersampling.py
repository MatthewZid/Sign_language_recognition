import pandas as pd

train_set_path = r'../train_dataset.csv'
MAXSIZE = 500

def undersample(path):
    """
    Undersample initial dataset to 500 examples for each class
    """
    dataset = pd.read_csv(path, header=None)

    frames = []
    for x in dataset[1].unique().tolist():
        df = dataset[dataset[1] == x]
        todrop = df.sample(n=(abs(len(df) - MAXSIZE)))
        undersampled = df.drop(todrop.index.tolist(), axis=0)
        frames.append(undersampled)

    conc = pd.concat(frames)
    conc = conc.reset_index(drop=True)
    result = conc.sample(frac=1).reset_index(drop=True)

    return result

result = undersample(train_set_path)

# write to csv
result.to_csv('../train_dataset_undersampled.csv', index=None, header=False)
print('OK')