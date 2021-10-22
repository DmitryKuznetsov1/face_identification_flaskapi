import numpy as np
import PIL.Image
import requests
import json
from app import init_params
URL = None


def load_image_file(file: str, mode: str = 'RGB') -> np.ndarray:
    """
    Загружает фото из файла и возвращает как numpy.ndarray
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def get_identification_report(ID: str, image_array: np.ndarray = None, image_path: str = None, URL = None) -> dict:
    """
    API для работы с сервером, Посылает GET запрос на идентификацию и возвращает dict (json)
    """
    if image_path:
        image_arr = load_image_file(image_path)
    else:
        image_arr = np.array(image_array, copy=True)
    image_json = json.dumps(image_arr.tolist())
    data = {'img_json': image_json, 'ID': ID}
    response = requests.get(URL, json=data)
    report = json.loads(response.content)
    return report


if __name__ == '__main__':
    config_file = "resources/config.json"
    params = init_params(config_file, "tolerance")
    URL = params["URL"]

    image_path = "samples/sample_img_0001.jpg"
    image_arr = load_image_file(image_path)
    ID = "0001"
    report = get_identification_report(ID, image_arr, URL=URL)
    print(report)
