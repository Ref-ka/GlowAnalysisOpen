import matplotlib.pyplot as plt
import os
import random
import torch


def visualize_predictions(model, dataset, epoch, device, save_dir="testing_data\\detector_train"):
    """
    Функция для визуальной проверки обучения детектора
    :param model: Детектор
    :param dataset: Обучающий набор данных
    :param epoch: Эпоха, для которой проверяются данные
    :param device: Устройство, на котором работает модель
    :param save_dir: Путь к директории для сохранения визуализаций
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    # Берем 4 рандомные изображения из тестового датасета
    N = 4
    idxs = random.sample(range(len(dataset)), min(N, len(dataset)))
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            img, gt_bbox = dataset[idx]
            input_img = img.unsqueeze(0).to(device)
            pred_bbox = model(input_img).cpu().squeeze().numpy()
            gt_bbox = gt_bbox.numpy()
            # Денормализуем изображения для визуализации
            img_vis = img.permute(1, 2, 0).cpu().numpy()
            img_vis = img_vis * [0.1556871, 0.18028949, 0.16393257] + [0.05502331, 0.08673652, 0.06147401]
            img_vis = (img_vis * 255).clip(0, 255).astype('uint8')
            # Рисуем границы
            fig, ax = plt.subplots(1)
            ax.imshow(img_vis)
            # Реальная разметка отмечена зеленым цветом
            x1, y1, x2, y2 = gt_bbox  # * 300
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none', label='GT')
            ax.add_patch(rect)
            # Предсказанная разметка отмечена красным цветом
            px1, py1, px2, py2 = pred_bbox  # * 300
            rect2 = plt.Rectangle((px1, py1), px2 - px1, py2 - py1, linewidth=2, edgecolor='r', facecolor='none',
                                  label='Pred')
            ax.add_patch(rect2)
            ax.set_title(f"Epoch {epoch}")
            ax.axis('off')
            plt.legend(handles=[rect, rect2], labels=['GT', 'Pred'])
            plt.savefig(os.path.join(save_dir, f"epoch{epoch}_sample{i}.png"))
            plt.close()
