import numpy as np
import pandas as pd
from time import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

MODEL_PATH = 'ckpt/model.0014-0.8818.h5'
DATASET_DIR = './dataset/test/'
BATCH_SIZE = 4
PROCESS_IMAGE_SIZE = (224, 224)
NUM_TEST_IMAGE = 506
NAME = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}


def load_testimage(image_size, dataset_dir, num_of_test_image, batch_size=32):
    """加载测试数据"""
    datagen = ImageDataGenerator()
    df = pd.DataFrame([str(i) + '.jpg' for i in range(num_of_test_image)], columns=['filename'])

    test_batches = datagen.flow_from_dataframe(dataframe=df,
                                               directory=dataset_dir,
                                               x_col='filename',
                                               y_col=None,
                                               target_size=image_size,
                                               interpolation='bicubic',
                                               class_mode=None,
                                               shuffle=False,
                                               batch_size=batch_size,
                                               validate_filenames=True)

    return test_batches


if __name__ == '__main__':
    # 加载测试数据
    test_batches = load_testimage(PROCESS_IMAGE_SIZE, DATASET_DIR, NUM_TEST_IMAGE, BATCH_SIZE)

    # 加载模型
    model = load_model(MODEL_PATH)
    name_dict = {value: key for key, value in NAME.items()}


    # 开始预测
    start = time()
    preds = model.predict(x=test_batches,
                          verbose=1)
    # 结束预测
    print(str(time() - start) + 's')

    preds_list = np.argmax(preds, axis=1)
    ans = []
    for i in preds_list:
        ans.append(name_dict[i])

    # 生成csv
    df = pd.DataFrame(ans)
    df.to_csv('ans.csv', index=True, header=False, encoding='utf-8')
