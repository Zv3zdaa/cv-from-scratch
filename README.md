# CV Algorithms From Scratch

Реализация базовых алгоритмов компьютерного зрения **с нуля на NumPy** —
без OpenCV и scipy. Цель проекта — понять математику, которая стоит
за `cv2.convolve`, `cv2.GaussianBlur`, `cv2.Canny` и `cv2.erode`.

## Что уже реализовано

- **2D-свёртка** (`cv_lib/convolution.py`) — методом скользящего окна,
  с zero-padding и произвольным нечётным ядром. Результат на внутренней
  области совпадает с `cv2.filter2D` (см. `tests/`).
  
  Замечание о корреляции vs свёртке. cv2.filter2D выполняет корреляцию, поэтому для сравнения с математической свёрткой ядро нужно перевернуть. PyTorch и TensorFlow, несмотря на название conv2d, тоже выполняют корреляцию — это конвенция глубокого обучения. Наш convolve2d реализует именно свёртку, поэтому совпадает с cv2.filter2D(img, -1, kernel[::-1, ::-1]).

## Roadmap

- [x] 2D-свёртка
- [ ] Фильтры Гаусса и Собеля
- [ ] Canny: NMS + двойной порог + hysteresis
- [ ] Морфология: erode / dilate / opening / closing

## Установка и запуск

\`\`\`bash
pip install -r requirements.txt

# демонстрация
python examples/demo_convolution.py

# проверка корректности против OpenCV
pytest tests/
\`\`\`

## Пример

Для матрицы 5×5 и ядра 3×3 `examples/demo_convolution.py` печатает
результат свёртки и сверяет его с `cv2.filter2D`. На внутренней области
значения совпадают.

## Лицензия

MIT