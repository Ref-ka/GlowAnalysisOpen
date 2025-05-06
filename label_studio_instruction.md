# Руководство по разметке данных с помощью label-studio
Важно отметить, что обработка лейблов производится строго отдельно для каждого набора данных по датам. Это сделано для того, чтобы при совершенной ошибке при разметке пострадала только часть данных.<br/>

## Шаг 1. Создание проекта
После старта label-studio откроется web-приложение, где сразу же после авторизации будет доступна функция создания проекта (кнопка Create)
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_1.png?raw=true)<br/>

Далее откроется окно создания проекта, нужно ввести название проекта.<br/>
Рекомендуется использовать названия проектов в таком виде, в котором это представлено на изображении
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_2.png?raw=true)<br/>

Затем нужно переключить на вкладку Data Import и загрузить файлы
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_3.png?raw=true)<br/>

Последний этап создания проекта - выбор сетапа для разметки, следуйте указаниям на изображении
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_4.png?raw=true)<br/>

Откроется страница настройки сетапа, нужно удалить метки Airplane и Car, вместо них добавить в поле слева метки glow и no_glow, далее нажать кнопку Add и затем Save
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_5.png?raw=true)<br/>

## Шаг 2. Разметка данных
Теперь можно приступать к разметке, выделяем все изображения и жмем Label Tasks
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_6.png?raw=true)<br/>

Во время разметки действует такой алгоритм:
1. Выбираем метку (glow - если есть свечение, no_glow - если нет свечения)
2. Если свечения нет - ставим прямоугольник в любом месте изображения, подтверждаем кнопкой Submit<br/>
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_7.png?raw=true)<br/>
3. Если свечение есть - выделяем прямоугольником область свечения. Важно выделять свечение не впритык, а оставлять немного места, как показано на изображении, далее также подтверждаем через Submit.<br/>
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_8.png?raw=true)<br/>
Спорный момент: на данный момент не совсем понятно, стоит ли отмечать как наличие совсем блеклых свечений, в этом плане возможны проблемы, так как матрица, полученная из такого изображения будет около нулевой, следовательно, она может быть просто вырожденной и модель её не обработает.<br/>
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_9.png?raw=true)<br/>

## Шаг 3. Экспорт разметки
После полной разметки набора данных нужно перейти на основную страницу проекта и нажать кнопку Export:<br/>
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_10.png?raw=true)<br/>
В открывшемся окне по-умолчанию будет выбран вариант JSON - он нам и нужен, просто жмем Export:<br/>
![alt text](https://github.com/Ref-ka/GlowAnalysis/blob/master/info_images/label_studio_11.png?raw=true)<br/>
Теперь json файл нужно положить в директорию, соответствующую набору данных, и переименовать в (labels.json). В примере использовался датасет 13.04.2025 для детектора, значит полученный файл должен находиться по пути training_data/13.04.2025/for_detector/labels.json

