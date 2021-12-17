from flask import Flask
from flask_restful import Api, Resource, reqparse
import json
from FaceIdentifier import FaceIdentifier
import numpy as np
from time import time
app = Flask(__name__)
api = Api(app)
face_identifier = None


def init_params(config_file: str, *to_eval) -> dict:
    """
    Инициализирует параметры из json файла, преобразует необходимые поля к их типам, напр. '0.7' -> float
    """
    with open(config_file, 'r') as json_params:
        params = json.load(json_params)
        params['tolerance'] = eval(params['tolerance'])
    return params


class Quote(Resource):
    """
    Предоставляет обработку GET запросов, вычисляет время запроса, кодирует входное изображение в json и возвращает
     результат идентификации
    """
    def get(self):
        request_time = time()
        parser = reqparse.RequestParser()
        parser.add_argument('img_json')
        parser.add_argument('ID')
        args = parser.parse_args()
        img_json = args['img_json']
        ID = args['ID']

        if (ID is not None) and (img_json is not None):
            img_arr = np.array(json.loads(img_json), dtype=np.uint8)
            result = face_identifier.identify(img_arr, ID, request_time=request_time)
            return result, 200
        return "Not found", 404


if __name__ == '__main__':
    config_file = "resources/config.json"
    database_map_file = "resources/source_map.json"

    params = init_params(config_file, "tolerance")
    database_map = init_params(database_map_file)

    face_identifier = FaceIdentifier(database_map, tolerance=params["tolerance"])
    api.add_resource(Quote, "/api/identify")
    app.run(debug=False)

