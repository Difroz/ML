# Rec_sys_topics_vs_cup
Задача о рекомендации контента пользователю.   
Пользуясь информацией о том, как пользователи взаимодействуют с контентом, составьте рекомендацию из 20 постов, которые в будущем привлекут максимум внимания конкретного пользователя.

Так как работал с ограниченными ресурсами (16gb оперативной памяти) некоторые действия разбиты на отдельные файлы.  

Решение задачи состояло из нескольких этапов:
* [Посмотреть на исходные данные](https://github.com/Difroz/ML_projects/blob/main/Rec_sys_topics_vk_cup/EDA.ipynb)
* [Подготовить данные для валидации](https://github.com/Difroz/ML_projects/blob/main/Rec_sys_topics_vk_cup/prepare_data.py)  
* Сформировать кандидатов ALS 
* [Обучить модель Catboost Classifier](https://github.com/Difroz/ML_projects/blob/main/Rec_sys_topics_vk_cup/CBC.ipynb)
