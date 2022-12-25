import shutil
import glob
from sklearn.model_selection import train_test_split

image = glob.glob('G:/tensorGO/dataset/human detection/valid/images/*jpg')

valid, test = train_test_split(image, test_size=0.3, random_state=42)

for i in test:
    shutil.move(i, i.replace('valid','test'))
    shutil.move((i.replace('images', 'labels')).replace('jpg', 'txt'), ((i.replace('valid','test').replace('images', 'labels').replace('jpg', 'txt'))))