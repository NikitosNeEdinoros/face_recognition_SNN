import os

PATH_NEG = './data/negative'
counter = 72
for directory in os.listdir('./dataset'):
    if directory != '.DS_Store':
        for file in os.listdir(os.path.join('./dataset', directory)):
            # os.rename(file, f'{counter}.png')
            PREV_FOLD = os.path.join('./dataset', directory, file)
            NEW_FOLD = os.path.join(PATH_NEG, file)
            os.replace(PREV_FOLD, NEW_FOLD)
            counter += 1


