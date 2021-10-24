import face_recognition as fr
import numpy as np
from typing import Tuple
from PIL import Image
import os
from time import time


class FaceIdentifier():
    def __init__(self, database_map: dict = None, tolerance: float = 0.7):
        self.database_map = database_map  # словарь, в котором ключ это ID, значение это список из [путь, должность,
        # id должности]
        self.successful_dir = "history/successful/"
        self.unsuccessful_dir = "history/unsuccessful/"
        self.tolerance = tolerance # порог, чем он больше, тем больше успешных идентификаций
        self.attempts_counter = {} # словарь, в котором ключ это ID, значение -- количество запросов этого ID (в том
        # числе не существующих); изменяется со временем, необходимо для уникального сохранения каждого запроса
        self.attempts_history = {} # словарь, в котором ключ это ID, значение -- список -- история времени запросов
        # Пример отчета (с возможными вариантами)
        # {
        #     "Идентификация": ["Ошибка", "Успешно"],
        #     "Причины": ["Другой человек в кадре",
        #                 "Более одного лица в кадре",
        #                 "Нет лиц в кадре",
        #                 "Идентификационный номер отсутствует в БД"],
        #     "Уверенность в схожести фотографии": "N%",
        #     "Должность": "CEO",
        #     "Идентификатор должности": "1",
        #     "Порог тончости": "0.7",
        #     "Отосланное фото сохранено в": "../history/unsuccessful/0001.jpeg"
        # }
        if "history" not in os.listdir():
            os.mkdir("history")
            os.mkdir(self.successful_dir)
            os.mkdir(self.unsuccessful_dir)

    def identify(self, input_image_arr: np.ndarray, ID: str, request_time: float) -> dict:
        are_similar = None
        error_msg = None
        confidence = None
        source_image_post = None
        source_image_post_id = None
        if ID not in self.database_map:
            error_msg = "Идентификационный номер отсутствует в БД"
        else:
            source_info = self.database_map[ID]
            source_image_path = source_info[0]
            source_image_post = source_info[1]
            source_image_post_id = source_info[2]
            source_image = fr.load_image_file(source_image_path)

            face_locations = fr.face_locations(input_image_arr)

            n_faces = len(face_locations)  # always >= 0
            if n_faces == 0:
                error_msg = "Нет лиц в кадре"
            elif n_faces > 1:
                error_msg = "Более одного лица в кадре"
            else:
                are_similar, confidence = self._check_similarity(input_image_arr, source_image,
                                                                 tolerance=self.tolerance)
        self.update_time_history(ID, request_time)
        where_to_save = self.save_image(are_similar, input_image_arr, provided_ID=ID)
        report = self._make_report(are_similar, error_msg, confidence, source_image_post, source_image_post_id,
                                   where_saved_path=where_to_save,
                                   tolerance=self.tolerance)

        return report

    def save_image(self, are_similar: bool, image_arr: np.ndarray, provided_ID: str) -> str:
        """
        Сохраняет отосланное фото в одну из директорий: successful/unsuccessful и вовращает путь до этого файла
        """
        count = self.get_number_of_attempts_and_inc(provided_ID)
        dir_to_save = (self.successful_dir if are_similar else self.unsuccessful_dir) + f"id{provided_ID}/"
        path_to_save = dir_to_save + str(count) + ".jpeg"
        im = Image.fromarray(image_arr)
        im.save(path_to_save)
        return path_to_save

    @staticmethod
    def _check_similarity(known_image: np.ndarray, unknown_image: np.ndarray, tolerance) -> Tuple[bool, str]:
        """
        Сравнивает схожесть двух фотографий, предоставленных как numpy.array. Возвращает результат сравнения, и
        'уверенность в сходести'
        """
        known_encoding = fr.face_encodings(known_image)[0]
        unknown_encoding = fr.face_encodings(unknown_image)[0]
        confidence = get_confidence([known_encoding], unknown_encoding)
        result = fr.compare_faces([known_encoding], unknown_encoding, tolerance=tolerance)
        return bool(result[0]), confidence

    @staticmethod
    def _make_report(are_similar: bool, error_msg: str, confidence: str, post: str, post_id: str, tolerance: float,
                     where_saved_path: str) -> dict:
        """
        Составляет и возвращает dict (json) ответ на запрос идентификации
        """
        report = {"Идентификация": "Успешно" if are_similar else "Ошибка",
                  "Причины": "Другой человек в кадре" if are_similar is False else error_msg,
                  "Уверенность в схожести фотографии": confidence if confidence else None, "Должность": post,
                  "Идентификатор должности": post_id, "Порог точности": tolerance,
                  "Отосланное фото сохранено в": where_saved_path}
        return report

    def get_number_of_attempts_and_inc(self, provided_ID):
        if provided_ID in self.attempts_counter:
            # increments
            self.attempts_counter[provided_ID] += 1
        else:
            self.attempts_counter[provided_ID] = 0
            os.mkdir(self.successful_dir + f"id{provided_ID}/")
            os.mkdir(self.unsuccessful_dir + f"id{provided_ID}/")

        return self.attempts_counter[provided_ID]

    def update_time_history(self, provided_ID: str, request_time: float) -> None:
        """
        Обновляет историю запросов для введенного ID
        """
        if provided_ID in self.attempts_history:
            self.attempts_history[provided_ID].append(request_time)
        else:
            self.attempts_history[provided_ID] = [request_time]
        print(f"{self.attempts_history=}")


def get_confidence(known_encodings: np.ndarray, unknown_encoding: np.ndarray) -> str:
    """
    Вычисляет уверенность системы в схожести фото. При сравнении двух фото, тем больше схожесть, чем меньше норма
     разности двух кодировок и именно это значение сравнивается с пороговым. Уверенность здесь интепретируется как
     = 1 - norm.
    """
    norm = np.linalg.norm(known_encodings - unknown_encoding, axis=1)[0]
    confidence = f"{round(1 - norm, 3) * 100}%"
    return confidence
